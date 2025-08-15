# ==== FINAL PIPELINE: Shuttle 5-fold, 3 models, full artifacts ====
!pip -q install ucimlrepo imbalanced-learn scipy

import warnings, numpy as np, pandas as pd, random, itertools, os
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, recall_score)
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_rel, wilcoxon

SEED = 42
np.random.seed(SEED); random.seed(SEED)
os.makedirs("outputs", exist_ok=True)

# --- Load ---
data = fetch_ucirepo(id=148)
X = data.data.features.copy()
y = data.data.targets.iloc[:,0].astype(int)
CLASSES = np.sort(y.unique())
RARE = [2,3,6,7]
EXCL1 = [2,3,4,5,6,7]

def metrics(y_true, y_pred):
    return dict(
        acc  = accuracy_score(y_true, y_pred),
        bacc = balanced_accuracy_score(y_true, y_pred),
        f1m  = f1_score(y_true, y_pred, average='macro'),
        f1w  = f1_score(y_true, y_pred, average='weighted'),
        rare = recall_score(y_true, y_pred, labels=RARE, average='macro', zero_division=0),
        f1x1 = f1_score(y_true, y_pred, labels=EXCL1, average='macro', zero_division=0)
    )

def summarize(name, recs):
    df = pd.DataFrame(recs)
    mean = df.mean().add_prefix(f"{name}__mean_")
    std  = df.std(ddof=1).add_prefix(f"{name}__std_")
    return df, mean, std

def plot_cm_norm(y_true_all, y_pred_all, labels, title, outpng):
    cm = confusion_matrix(y_true_all, y_pred_all, labels=labels, normalize='true')
    plt.figure(figsize=(7,6))
    plt.imshow(cm, interpolation='nearest')
    plt.title(title); plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels, rotation=45); plt.yticks(ticks, labels)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]:.2f}", ha="center",
                 color="white" if cm[i, j] > 0.5 else "black")
        # no colors specified per instructions
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()

def plot_recall_bars(per_class_recall, models, labels, outpng):
    # per_class_recall: dict[model] -> array of recall per label (aligned with labels)
    x = np.arange(len(labels))
    width = 0.27
    plt.figure(figsize=(9,4.8))
    for k, m in enumerate(models):
        plt.bar(x + k*width, per_class_recall[m], width=width, label=m)
    plt.xticks(x + width, labels)
    plt.ylabel("Recall")
    plt.title("Per-class recall (aggregated over folds)")
    plt.legend()
    plt.tight_layout(); plt.savefig(outpng, dpi=300); plt.close()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

rec_gb, rec_sm, rec_rf = [], [], []
# store per-fold metrics for tests
f1m_gb, f1m_sm, f1m_rf = [], [], []
bacc_gb, bacc_sm, bacc_rf = [], [], []

# aggregate predictions for CM/recall bars
y_all, yhat_gb_all, yhat_sm_all, yhat_rf_all = [], [], [], []

for tr, te in skf.split(X, y):
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    y_tr, y_te = y.iloc[tr], y.iloc[te]

    # class weights
    cls = np.sort(np.unique(y_tr))
    cw = compute_class_weight("balanced", classes=cls, y=y_tr)
    cw_dict = {c:w for c,w in zip(cls, cw)}
    sw_tr = compute_sample_weight(class_weight=cw_dict, y=y_tr)

    # GB base (cost-sensitive)
    gb = GradientBoostingClassifier(n_estimators=320, learning_rate=0.09, max_depth=3, random_state=SEED)
    gb.fit(X_tr, y_tr, sample_weight=sw_tr)
    p_gb = gb.predict(X_te)
    rec = metrics(y_te, p_gb); rec_gb.append(rec)
    f1m_gb.append(rec["f1m"]); bacc_gb.append(rec["bacc"])

    # GB + SMOTE(k=1)
    sm = SMOTE(sampling_strategy='not majority', k_neighbors=1, random_state=SEED)
    X_sm, y_sm = sm.fit_resample(X_tr, y_tr)
    gb_sm = GradientBoostingClassifier(n_estimators=380, learning_rate=0.08, max_depth=3, random_state=SEED)
    gb_sm.fit(X_sm, y_sm)
    p_sm = gb_sm.predict(X_te)
    rec = metrics(y_te, p_sm); rec_sm.append(rec)
    f1m_sm.append(rec["f1m"]); bacc_sm.append(rec["bacc"])

    # RF + class_weight
    rf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1, class_weight=cw_dict)
    rf.fit(X_tr, y_tr)
    p_rf = rf.predict(X_te)
    rec = metrics(y_te, p_rf); rec_rf.append(rec)
    f1m_rf.append(rec["f1m"]); bacc_rf.append(rec["bacc"])

    # collect for plots
    y_all.append(y_te.to_numpy())
    yhat_gb_all.append(p_gb); yhat_sm_all.append(p_sm); yhat_rf_all.append(p_rf)

# concat predictions
y_all = np.concatenate(y_all)
yhat_gb_all = np.concatenate(yhat_gb_all)
yhat_sm_all = np.concatenate(yhat_sm_all)
yhat_rf_all = np.concatenate(yhat_rf_all)

# summaries
df_gb, m_gb, s_gb = summarize("GB_base", rec_gb)
df_sm, m_sm, s_sm = summarize("GB_SMOTE", rec_sm)
df_rf, m_rf, s_rf = summarize("RF_cw", rec_rf)

summary = pd.concat([m_gb, s_gb, m_sm, s_sm, m_rf, s_rf], axis=0)
summary.to_csv("outputs/summary_mean_std.csv", index=True)

print("\n=== SUMMARY (mean±std to CSV) ===\n", summary.round(4))

# significance tests (paired, per fold)
def pair_tests(a, b, label):
    t = ttest_rel(a, b)
    try:
        w = wilcoxon(a, b, zero_method="wilcox", correction=False, alternative='two-sided')
    except ValueError:
        w = ("wilcoxon_failed",)
    print(f"{label} | t-test: p={getattr(t, 'pvalue', None)} | wilcoxon: {w}")

pair_tests(f1m_gb, f1m_sm, "Macro-F1: GB_base vs GB_SMOTE")
pair_tests(f1m_gb, f1m_rf, "Macro-F1: GB_base vs RF_cw")
pair_tests(bacc_gb, bacc_sm, "BalancedAcc: GB_base vs GB_SMOTE")
pair_tests(bacc_gb, bacc_rf, "BalancedAcc: GB_base vs RF_cw")

# confusion matrix (normalized) for GB_base
plot_cm_norm(y_all, yhat_gb_all, CLASSES, "Normalized Confusion Matrix — GB_base", "outputs/cm_gb_base.png")

# per-class recall bars
def per_class_recall(y_true, y_pred, labels):
    r = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return r

recall_gb = per_class_recall(y_all, yhat_gb_all, CLASSES)
recall_sm = per_class_recall(y_all, yhat_sm_all, CLASSES)
recall_rf = per_class_recall(y_all, yhat_rf_all, CLASSES)

per_class = {
    "GB_base": recall_gb,
    "GB_SMOTE": recall_sm,
    "RF_cw": recall_rf
}
plot_recall_bars(per_class, ["GB_base","GB_SMOTE","RF_cw"], CLASSES, "outputs/recall_per_class.png")

# save fold-level metrics for reproducibility
pd.DataFrame(rec_gb).to_csv("outputs/folds_gb_base.csv", index=False)
pd.DataFrame(rec_sm).to_csv("outputs/folds_gb_smote.csv", index=False)
pd.DataFrame(rec_rf).to_csv("outputs/folds_rf_cw.csv", index=False)

print("\nArtifacts saved in ./outputs/:")
print("- summary_mean_std.csv")
print("- folds_*.csv (per-fold metrics)")
print("- cm_gb_base.png (normalized)")
print("- recall_per_class.png")