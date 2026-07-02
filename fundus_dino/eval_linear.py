# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from PIL import Image
import utils
import vision_transformer as vits
from collections import Counter
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, average_precision_score,
    hamming_loss, jaccard_score, recall_score, precision_score, cohen_kappa_score, roc_curve
)
from pycm import ConfusionMatrix
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import scipy.stats


# =====================================================================
# DeLong's method for AUROC confidence intervals (analytic, no resampling)
# Standard fast-DeLong implementation (Sun & Xu, 2014), used widely in
# medical imaging papers for reporting AUROC 95% CIs.
# =====================================================================
def _compute_midrank(x):
    """Computes midranks, used internally by fast DeLong."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    predictions_sorted_transposed: [1, N] array of scores, sorted so that
        all positive-class examples (label==1) come first.
    label_1_count: number of positive examples (m).
    Returns (auc, auc_var) where auc_var is the DeLong variance estimate.
    """
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs[0], delongcov


def delong_roc_ci(y_true, y_pred, alpha=0.95):
    """
    Computes AUROC and its (alpha*100)% CI via DeLong's method.

    y_true: array-like of 0/1 ground truth labels.
    y_pred: array-like of predicted probabilities/scores for the positive class.
    Returns: (auc, ci_lower, ci_upper)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    assert set(np.unique(y_true)).issubset({0, 1}), "y_true must be binary (0/1)"

    order = (-y_true).argsort()  # positives first
    label_1_count = int(y_true.sum())

    predictions_sorted_transposed = y_pred[np.newaxis, order]
    auc, auc_cov = _fast_delong(predictions_sorted_transposed, label_1_count)
    auc_std = np.sqrt(auc_cov)

    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = scipy.stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
    ci = np.clip(ci, 0, 1)

    return float(auc), float(ci[0]), float(ci[1])
# =====================================================================

def eval_linear(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    linear_classifier = LinearClassifier(embed_dim, num_labels=args.num_labels)
    linear_classifier = linear_classifier.cuda()
    linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

    # ============ preparing data ... ============
    val_transform = pth_transforms.Compose([
        pth_transforms.Resize(224 , interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if args.evaluate:
        utils.load_linear_weights_local(args, linear_classifier)
        results = []
        dataset_names = ["REFUGE", "PAPILA", "FIVES", "ORIGA", "CHAKSU", "ACRIMA", "LAG"]
        for path in dataset_names:
            dataset_test = datasets.ImageFolder(os.path.join(args.data_path, path), transform=val_transform)
            loader = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=args.batch_size_per_gpu,
                num_workers=args.num_workers,
                pin_memory=True,
                drop_last=False,
            )
            print(f"Evaluating on {len(loader.dataset)} images...")
            results.append(evaluate(loader, model,linear_classifier, args.n_last_blocks, args.avgpool_patchtokens))

        # ---- Summary across all datasets, with 95% CIs (DeLong) ----
        print("\nResults (ROC AUC, 95% CI via DeLong's method):")
        for name, (auc, ci_lo, ci_hi) in zip(dataset_names, results):
            print(f"  {name}: {auc:.4f} (95% CI: {ci_lo:.4f}-{ci_hi:.4f})")

        aucs_only = [r[0] for r in results]
        print(f"\nMean AUROC across {len(dataset_names)} datasets: "
              f"{np.mean(aucs_only):.4f} ± {np.std(aucs_only):.4f}")
        return

    train_transform = pth_transforms.Compose([
        pth_transforms.Resize(224, interpolation=pth_transforms.InterpolationMode.BICUBIC),
        pth_transforms.CenterCrop(224),
        pth_transforms.RandomRotation(
            degrees=30,
            interpolation=pth_transforms.InterpolationMode.BICUBIC
        ),
        pth_transforms.RandomHorizontalFlip(p=0.5),
        pth_transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02
        ),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225)
        ),
        ])
    dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    labels = [label for _, label in dataset_train.samples]
    class_counts = Counter(labels)

    num_classes = len(class_counts)
    total_samples = len(labels)

    weights = []
    for i in range(num_classes):
        weights.append(total_samples / (num_classes * class_counts[i]))

    weights = torch.tensor(weights, dtype=torch.float).cuda()
    sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0, # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    # Optionally resume from a checkpoint
    to_restore = {"epoch": 0, "best_acc": 0.}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth.tar"),
        run_variables=to_restore,
        state_dict=linear_classifier,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    start_epoch = to_restore["epoch"]
    best_acc = to_restore["best_acc"]

    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, weights)
        scheduler.step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            roc_auc, _, _ = evaluate(val_loader, model, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)
            best_auc = max(best_auc, roc_auc)
            print(f'Max AUROC so far: {best_auc:.2f}%')
            log_stats = {**{k: v for k, v in log_stats.items()},
                        'test_auc': round(roc_auc, 4),
                        'best_auc': round(best_auc, 4)}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            save_dict = {
                "epoch": epoch + 1,
                "state_dict": linear_classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_acc": best_acc,
            }
            torch.save(save_dict, os.path.join(args.output_dir, "checkpoint.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))
    evaluate(val_loader, model,linear_classifier, args.n_last_blocks, args.avgpool_patchtokens)

def train(model, linear_classifier, optimizer, loader, epoch, n, avgpool, weights):
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)

        loss = nn.CrossEntropyLoss(weight=weights)(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, linear_classifier, n, avgpool, device="cuda", mode='test', num_class=2):
    """Evaluate the model.

    Returns:
        (roc_auc, ci_lower, ci_upper): AUROC and its 95% CI computed via
        DeLong's method.
    """
    model.eval()
    linear_classifier.eval()
    
    true_onehot, true_labels = [], []
    all_glaucoma_probs = []  # collect raw probs first, threshold later

    # ── Pass 1: collect all probabilities ──────────────────────────────────
    for batch in data_loader:
        images  = batch[0].to(device, non_blocking=True)
        target  = batch[-1].to(device, non_blocking=True)
        target_onehot = F.one_hot(target.to(torch.int64), num_classes=num_class)

        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(images, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat(
                        (output.unsqueeze(-1),
                         torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)),
                        dim=-1
                    )
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(images)

        output        = linear_classifier(output)
        output_       = nn.Softmax(dim=1)(output)          # [B, 2]
        glaucoma_prob = output_[:, 1]                      # P(glaucoma)

        true_onehot.extend(target_onehot.cpu().numpy())
        true_labels.extend(target.cpu().numpy())
        all_glaucoma_probs.extend(glaucoma_prob.detach().cpu().numpy())

    # ── Find optimal threshold AFTER seeing all predictions ────────────────
    fpr, tpr, thresholds = roc_curve(true_labels, all_glaucoma_probs)

    # Youden's J — best balance of sensitivity + specificity
    J            = tpr - fpr
    optimal_idx  = J.argmax()
    optimal_threshold = thresholds[optimal_idx]
    print(f'Optimal threshold (Youden): {optimal_threshold:.4f}')

    true_labels_np = np.array(true_labels)
    true_onehot_np = np.array(true_onehot)
    all_glaucoma_probs_np = np.array(all_glaucoma_probs)

    for optimal_threshold in [0.5, optimal_threshold]:
        # ── Pass 2: apply threshold to get final labels ─────────────────────────
        pred_labels  = (all_glaucoma_probs_np >= optimal_threshold).astype(int)
        pred_onehot  = np.eye(num_class)[pred_labels]          # one-hot from labels

        # softmax array needed for average_precision
        # reconstruct as [N, 2] from glaucoma probs
        pred_softmax_np = np.stack(
            [1 - all_glaucoma_probs_np, all_glaucoma_probs_np], axis=1
        )

        # ── Metrics ────────────────────────────────────────────────────────────
        accuracy          = accuracy_score(true_labels_np, pred_labels)
        hamming           = hamming_loss(true_onehot_np, pred_onehot)
        jaccard           = jaccard_score(true_onehot_np, pred_onehot, average='macro')
        average_precision = average_precision_score(true_onehot_np, pred_softmax_np, average='macro')
        kappa             = cohen_kappa_score(true_labels_np, pred_labels)

        precision = precision_score(true_labels_np, pred_labels, pos_label=1, zero_division=0)
        recall    = recall_score   (true_labels_np, pred_labels, pos_label=1, zero_division=0)
        f1        = f1_score       (true_labels_np, pred_labels, pos_label=1, zero_division=0)
        roc_auc   = roc_auc_score  (true_labels_np, all_glaucoma_probs_np)

        score = (f1 + roc_auc + kappa) / 3

        print(f'Threshold: {optimal_threshold:.4f}')
        print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, ROC AUC: {roc_auc:.4f}, '
            f'Hamming Loss: {hamming:.4f},\n'
            f'Jaccard Score: {jaccard:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f},\n'
            f'Average Precision: {average_precision:.4f}, Kappa: {kappa:.4f}, Score: {score:.4f}')

    # ── 95% CI for AUROC via DeLong's method ────────────────────────────────
    auc_delong, ci_lower, ci_upper = delong_roc_ci(true_labels_np, all_glaucoma_probs_np, alpha=0.95)
    print(f'AUROC: {auc_delong:.4f} (95% CI: {ci_lower:.4f}-{ci_upper:.4f}) [DeLong]')

    if mode == 'test':
        cm = ConfusionMatrix(actual_vector=true_labels_np.tolist(),
                             predict_vector=pred_labels.tolist())
        cm.plot(cmap=plt.cm.Blues, number_label=True, normalized=False, plot_lib="matplotlib")
        plt.savefig(os.path.join(args.output_dir, 'confusion_matrix_test.jpg'),
                    dpi=600, bbox_inches='tight')
    
    return roc_auc, ci_lower, ci_upper


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, n, avgpool):
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output, target, topk=(1,))

        batch_size = inp.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
            
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class MLPClassifier(nn.Module):
    """MLP classifier with two hidden layers on top of frozen features"""
    def __init__(self, dim, num_labels=1000, hidden_dim=1024):
        super(MLPClassifier, self).__init__()
        self.num_labels = num_labels

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_labels)
        )

        # Initialize weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                layer.weight.data.normal_(mean=0.0, std=0.01)
                layer.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # mlp layers
        return self.mlp(x)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default = 1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='./weights/task_dino/checkpoint.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=50, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.0001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=256, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='./dataset', type=str)
    parser.add_argument('--num_workers', default=6, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default = 1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./output_dir/linear_probe/task_dino", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default= 2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    args = parser.parse_args()
    eval_linear(args)