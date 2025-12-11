import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import os, sys, math
sys.path.append('./Utils/')

from ECGdataset import ECGDataset
from net1d import Net1D

class UncertaintyWeighter(nn.Module):
    def __init__(self, num_tasks: int):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_tasks, dtype=torch.float))

    def precisions(self):
        return torch.exp(-self.log_vars)

    def reg_term(self):
        return self.log_vars.sum()


def pcgrad_step(model, task_losses, optimizer):
    params = [p for p in model.parameters() if p.requires_grad]
    per_task_grads = []
    for li in task_losses:
        optimizer.zero_grad(set_to_none=True)
        li.backward(retain_graph=True)
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
        per_task_grads.append(grads)

    for i in range(len(per_task_grads)):
        for j in range(i + 1, len(per_task_grads)):
            dot = torch.tensor(0., device=params[0].device)
            norm_j_sq = torch.tensor(0., device=params[0].device)
            for gi, gj in zip(per_task_grads[i], per_task_grads[j]):
                dot += (gi * gj).sum()
                norm_j_sq += (gj * gj).sum()
            if dot < 0 and norm_j_sq > 0:
                coeff = dot / (norm_j_sq + 1e-12)
                for k in range(len(per_task_grads[i])):
                    per_task_grads[i][k] -= coeff * per_task_grads[j][k]

    optimizer.zero_grad(set_to_none=True)
    for p in params:
        p.grad = torch.zeros_like(p)

    for grads in per_task_grads:
        for p, g in zip(params, grads):
            p.grad.add_(g / len(per_task_grads))

    optimizer.step()


def build_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def my_roc_curve(df, model_path):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import os
    import pandas as pd

    LABELS = [
        '右冠状动脉主干_诊断结果', 
        '左冠状动脉主干_诊断结果', 
        '左前降支_诊断结果', 
        '左回旋支_诊断结果'
    ]
    PROBS = [
        '右冠状动脉主干_prob', 
        '左冠状动脉主干_prob', 
        '左前降支_prob', 
        '左回旋支_prob'
    ]
    CLASS_NAMES = ['RCA', 'LM', 'LAD', 'LCx']

    NEG_STR_SET = {"轻度狭窄", "中度狭窄", "未见明显狭窄"}

    def binarize(x):
        if pd.isna(x):
            return 0  
        x = str(x).strip()
        return 0 if x in NEG_STR_SET else 1

    plt.figure(figsize=(8, 6))

    for i in range(len(LABELS)):
        if LABELS[i] not in df or PROBS[i] not in df:
            continue

        temp = df[[LABELS[i], PROBS[i]]].dropna()
        y = temp[LABELS[i]].apply(binarize)
        s = temp[PROBS[i]]

        if len(np.unique(y)) < 2:
            continue

        fpr, tpr, _ = roc_curve(y, s)
        roc_auc = auc(fpr, tpr)

        plt.plot(
            fpr, tpr, lw=2,
            label=f"{CLASS_NAMES[i]} (AUC={roc_auc:.3f})"
        )

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Coronary Arteries)")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(model_path, "roc_curve_oof.png")
    plt.savefig(save_path, dpi=300)
    plt.close()




def run_fold_training(train_df, test_df, fold_num, config):
    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"--- Fold {fold_num + 1} ---")

    fold_name = f'fold_{fold_num + 1}'
    MODEL_PATH_FOLD = os.path.join(config['MODEL_PATH'], fold_name)
    LOG_DIR_FOLD = os.path.join(config['TENSORBOARD_LOG_DIR'], fold_name)
    os.makedirs(MODEL_PATH_FOLD, exist_ok=True)
    os.makedirs(LOG_DIR_FOLD, exist_ok=True)
    MODEL_SAVE_PATH = os.path.join(MODEL_PATH_FOLD, "best_model_macro_auc.pth")

    train_dataset = ECGDataset(train_df, use_augment=True, total_epochs=config['EPOCHS'], strict_level="yes")
    test_dataset = ECGDataset(test_df, use_augment=False, strict_level="yes")

    train_loader = DataLoader(train_dataset, batch_size=config['BATCH_SIZE'], shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
                             num_workers=2, pin_memory=True)

    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64,160,160,400,400,1024,1024],
        m_blocks_list=[2,2,2,3,3,4,4],
        kernel_size=16, stride=2, groups_width=16,
        verbose=False, use_bn=True, use_do=True, n_classes=4
    )

    ckpt = torch.load(config['PRETRAINED_MODEL_PATH'], map_location=device)
    state_dict = {k: v for k, v in ckpt.items() if not k.startswith('dense.')}
    model.load_state_dict(state_dict, strict=False)
    model.dense = nn.Linear(model.dense.in_features, 4)
    model = model.to(device)

    uw = UncertaintyWeighter(4).to(device)
    bce_none = nn.BCEWithLogitsLoss(reduction='none')
    optimizer = optim.AdamW(list(model.parameters()) + list(uw.parameters()), lr=config['LEARNING_RATE'], weight_decay=5e-4)

    total_steps = len(train_loader) * config['EPOCHS']
    scheduler = build_warmup_cosine_scheduler(optimizer, int(0.1 * total_steps), total_steps)
    best_macro_auc = 0.0

    writer = SummaryWriter(LOG_DIR_FOLD)

    for epoch in range(config['EPOCHS']):
        train_dataset.set_epoch(epoch) 
        model.train()
        uw.train()
        running_loss = 0.0

        for signals_origin, signals_aug, labels in train_loader:
            signals_origin = signals_origin.to(device)
            signals_aug = signals_aug.to(device)
            labels = labels.to(device)
            signals = torch.cat([signals_origin, signals_aug], dim=0)
            labels = torch.cat([labels, labels], dim=0)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(signals)
            per_elem = bce_none(outputs, labels)
            task_losses = per_elem.mean(dim=0)
            precision = uw.precisions()
            weighted_losses = [precision[i] * task_losses[i] for i in range(4)]
            pcgrad_step(model, weighted_losses, optimizer)

            optimizer.zero_grad(set_to_none=True)
            loss_vars = torch.sum(torch.exp(-uw.log_vars) * task_losses.detach()) + uw.reg_term()
            loss_vars.backward()
            optimizer.step()
            scheduler.step()
            running_loss += task_losses.mean().item()

        epoch_loss = running_loss / len(train_loader)
        
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        print(f"[Fold {fold_num+1}][Epoch {epoch+1}] TrainLoss={epoch_loss:.4f} | log_vars={uw.log_vars.data.cpu().numpy()}")

        model.eval(); uw.eval()
        val_loss = 0.0
        all_labels, all_probs = [], []
        with torch.no_grad():
            for signals, _, labels in test_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                per_elem = bce_none(outputs, labels)
                val_loss += per_elem.mean().item()
                probs = torch.sigmoid(outputs)
                all_labels.append(labels.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
        val_loss /= len(test_loader)

        all_labels = np.concatenate(all_labels)
        all_probs = np.concatenate(all_probs)

        macro_auc = 0.0; n_valid = 0
        for i in range(all_labels.shape[1]):
            if len(np.unique(all_labels[:, i])) > 1:
                auc_i = roc_auc_score(all_labels[:, i], all_probs[:, i])
                writer.add_scalar(f"AUC/class_{i}", auc_i, epoch)
                macro_auc += auc_i; n_valid += 1
        macro_auc = macro_auc / n_valid if n_valid > 0 else 0
        writer.add_scalar("Metric/macro_auc", macro_auc, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        print(f"[Fold {fold_num+1}][Epoch {epoch+1}] ValLoss={val_loss:.4f}, MacroAUC={macro_auc:.4f}")

        if macro_auc > best_macro_auc:
            best_macro_auc = macro_auc
            torch.save({'model': model.state_dict(), 'uw': uw.state_dict()}, MODEL_SAVE_PATH)
            print(f"*** Fold{fold_num+1} 保存最佳模型，MacroAUC={best_macro_auc:.4f} ***")

    writer.close()
    ckpt_best = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt_best['model'])
    uw.load_state_dict(ckpt_best['uw'])
    model.eval()

    final_probs = []
    final_labels = []
    test_loader_final = DataLoader(test_dataset, batch_size=config['BATCH_SIZE'], shuffle=False,
                                   num_workers=2, pin_memory=True)

    with torch.no_grad():
        for signals, _, labels in test_loader_final:
            signals = signals.to(device, non_blocking=True)
            probs = torch.sigmoid(model(signals))
            final_probs.append(probs.cpu().numpy())
            final_labels.append(labels.cpu().numpy())

    final_probs = np.concatenate(final_probs)
    final_labels = np.concatenate(final_labels)
    
    prob_df = pd.DataFrame(final_probs, columns=[f'{col}_prob' for col in ['右冠状动脉主干','左冠状动脉主干','左前降支','左回旋支']])
    label_df = pd.DataFrame(final_labels, columns=['右冠状动脉主干_label','左冠状动脉主干_label','左前降支_label','左回旋支_label'])

    result_df = pd.concat([test_df.reset_index(drop=True), label_df, prob_df], axis=1)
    fold_csv_path = os.path.join(MODEL_PATH_FOLD, f'predictions_{fold_name}_best.csv')
    result_df.to_csv(fold_csv_path, index=False, encoding='utf_8_sig')
    
    my_roc_curve(result_df, MODEL_PATH_FOLD)


    print(f"Fold {fold_num+1} 完成训练。最佳 MacroAUC={best_macro_auc:.4f}")


def run_kfold_pipeline(config): 
    torch.manual_seed(config['RANDOM_SEED'])
    np.random.seed(config['RANDOM_SEED'])

    full_df = pd.read_csv(config['CSV_PATH'])

    LABELS = [
        '右冠状动脉主干_诊断结果',
        '左冠状动脉主干_诊断结果',
        '左前降支_诊断结果',
        '左回旋支_诊断结果'
    ]

    NEG_STR_SET = {"未见明显狭窄", "轻度狭窄", "中度狭窄"}

    def binarize(x):
        return 0 if x in NEG_STR_SET else 1

    y_stratify = full_df[LABELS].applymap(binarize).astype(str).agg(''.join, axis=1)

    groups = full_df[config['ID_COLUMN_NAME']]
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['RANDOM_SEED'])


    oof_predictions = [] 

    for fold_num, (train_idx, test_idx) in enumerate(sgkf.split(full_df, y_stratify, groups)):
        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        test_df  = full_df.iloc[test_idx].reset_index(drop=True)
        
        run_fold_training(train_df, test_df, fold_num, config)
        
        fold_name = f'fold_{fold_num + 1}'
        fold_csv_path = os.path.join(config['MODEL_PATH'], fold_name, f'predictions_{fold_name}_best.csv')
        if os.path.exists(fold_csv_path):
             oof_predictions.append(pd.read_csv(fold_csv_path))

    if oof_predictions:
        oof_df = pd.concat(oof_predictions).sort_index()
        oof_df.to_csv(os.path.join(config['MODEL_PATH'], 'oof_predictions_all_folds.csv'), index=False, encoding='utf_8_sig')
        my_roc_curve(oof_df, config['MODEL_PATH'])


if __name__ == "__main__":
    my_config = {
        'CSV_PATH': './data/sample.csv',
        'ID_COLUMN_NAME': '患者编号',
        'BATCH_SIZE': 64, 
        'EPOCHS': 100,
        'LEARNING_RATE': 3e-5,
        'RANDOM_SEED': 42,
        'MODEL_PATH': './outputs/checkpoint', 
        'TENSORBOARD_LOG_DIR': './outputs/logs', 
        'PRETRAINED_MODEL_PATH': './1m-epoch15.pth',
    }
    
    os.makedirs(my_config['MODEL_PATH'], exist_ok=True)
    os.makedirs(my_config['TENSORBOARD_LOG_DIR'], exist_ok=True)

    run_kfold_pipeline(my_config)
