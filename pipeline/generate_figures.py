"""
Generate publication-quality, 600 DPI IEEE paper figures in results/figures/:
- Figure 1: End-to-End System Pipeline Flowchart
- Figure 2: TMG-GAN Neural Architecture (without Algorithm 2 box)
- Figures 3-5: Copied from evaluation results (Benign FPR, Recall by Class, Confusion Matrix)
- Figure 6: High-resolution (600 DPI) 2D t-SNE Projection with horizontal top legend
"""

import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_FIGS = ROOT / "results" / "figures"
PAPER_FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 9

def draw_system_pipeline():
    """Figure 1: End-to-End System Pipeline Flowchart."""
    fig, ax = plt.subplots(figsize=(13.5, 4.4), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 48)

    c_blue = '#1E3A8A'     # Preprocessing (Deep Blue)
    c_slate = '#334155'    # Reshaping (Slate)
    c_teal = '#0F766E'     # GAN Training (Teal)
    c_purple = '#6D28D9'   # Quality Filter (Purple)
    c_amber = '#B45309'    # Evaluation (Amber/Orange)

    def draw_box(x, y, w, h, step_num, title, lines, color, bg='#FFFFFF'):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                              edgecolor=color, facecolor=bg, linewidth=1.6)
        ax.add_patch(rect)
        
        banner_h = 5.5
        header = FancyBboxPatch((x, y + h - banner_h), w, banner_h, 
                                boxstyle="round,pad=0.1,rounding_size=0.4",
                                edgecolor=color, facecolor=color, linewidth=0)
        ax.add_patch(header)
        
        header_text = f"{step_num}. {title}"
        ax.text(x + w/2, y + h - banner_h/2, header_text, ha='center', va='center', 
                color='white', fontsize=8.2, fontweight='bold')
        
        body_y = y + (h - banner_h) / 2
        ax.text(x + w/2, body_y, "\n".join(lines), ha='center', va='center', 
                color='#0F172A', fontsize=7.2, multialignment='center', linespacing=1.35)

    def draw_sleek_arrow(x1, y1, x2, y2, label=""):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(facecolor='#334155', edgecolor='#334155',
                                    arrowstyle='-|>', lw=1.8, mutation_scale=15))
        if label:
            # Clean floating text above arrow with comfortable margin
            ax.text((x1 + x2)/2, y1 + 1.8, label, ha='center', va='bottom',
                    fontsize=6.8, color='#475569', fontweight='bold')

    # Top headers
    ax.text(52.5, 45.5, "End-to-End TMG-GAN Intrusion Detection & Evaluation Pipeline",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#0F172A')
    ax.text(52.5, 42.8, "Strict zero-leakage workflow from raw PCAP flow records to large-scale benchmark evaluation",
            ha='center', va='center', fontsize=8.2, fontstyle='italic', color='#475569')

    box_y = 3.5
    box_h = 35.0
    box_w = 14.2
    gap = 6.2

    # Box 1: Preprocessing
    x1 = 2.5
    draw_box(x1, box_y, box_w, box_h, "1", "Preprocessing",
             ["CICIDS2017 Flows",
              "(2.83M Raw Records)",
              r"$\downarrow$",
              "Clean Inf & NaNs",
              "78 Numerical Features",
              r"$\downarrow$",
              "70/30 Stratified Split",
              "Fit Scaler on Train",
              "[0, 1] Min-Max Norm"],
             c_blue, '#F0F9FF')

    # Arrow 1 -> 2
    x2 = x1 + box_w + gap
    draw_sleek_arrow(x1 + box_w + 0.4, box_y + box_h/2, x2 - 0.4, box_y + box_h/2)

    # Box 2: Spatial Reshaping
    draw_box(x2, box_y, box_w, box_h, "2", "Feature Reshape",
             ["1D Flow Vector",
              r"$\mathbf{x} \in [0, 1]^{78}$",
              r"$\downarrow$",
              "Spatial Matrix Mapping",
              r"$\mathbf{X} \in \mathbb{R}^{13 \times 6 \times 1}$",
              r"$\downarrow$",
              "Enables 2D Convolutions",
              "Feature Weight Sharing",
              r"Latent $z \sim \mathcal{N}(0, I)$"],
             c_slate, '#F8FAFC')

    # Arrow 2 -> 3
    x3 = x2 + box_w + gap
    draw_sleek_arrow(x2 + box_w + 0.4, box_y + box_h/2, x3 - 0.4, box_y + box_h/2)

    # Box 3: TMG-GAN Training
    draw_box(x3, box_y, box_w, box_h, "3", "TMG-GAN Train",
             ["Multi-Generators",
              r"$(G_1, \dots, G_K)$",
              "Class-specific MLPs",
              r"$\downarrow$",
              "Shared Conv2D Trunk",
              "Spectral Normalization",
              r"$\downarrow$",
              "Tied Discriminator $D$",
              "& Auxiliary Classifier $C$"],
             c_teal, '#F0FDFA')

    # Arrow 3 -> 4
    x4 = x3 + box_w + gap
    draw_sleek_arrow(x3 + box_w + 0.4, box_y + box_h/2, x4 - 0.4, box_y + box_h/2)

    # Box 4: Quality Gate
    draw_box(x4, box_y, box_w, box_h, "4", "Quality Filter",
             ["Algorithm 2 Gate:",
              "Confidence Verification:",
              r"$C(\hat{\mathbf{x}})_k \geq 0.50$",
              "Realism Verification:",
              r"$D(\hat{\mathbf{x}}) \geq \tau_D$",
              r"$\downarrow$",
              "Rejects Noise Flows",
              "98.1% Pass Rate",
              "+13,059 per Class"],
             c_purple, '#FAF5FF')

    # Arrow 4 -> 5
    x5 = x4 + box_w + gap
    draw_sleek_arrow(x4 + box_w + 0.4, box_y + box_h/2, x5 - 0.4, box_y + box_h/2)

    # Box 5: Benchmark Evaluation
    draw_box(x5, box_y, box_w + 1.2, box_h, "5", "Evaluation",
             ["Downstream 3-Seed DNN",
              "Identical Architecture",
              r"$\downarrow$",
              "3 Resampling Regimes:",
              r"$\bullet$ Baseline (145k)",
              r"$\bullet$ SMOTE (210k)",
              r"$\bullet$ TMG-GAN (210k)",
              r"$\downarrow$",
              "710,976 Test Flows",
              "FPR & Recall Analysis"],
             c_amber, '#FFFBEB')

    plt.tight_layout()
    out_path = PAPER_FIGS / "fig1_system_pipeline.png"
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1 (600 DPI): {out_path}")


def draw_model_architecture():
    """Figure 2: TMG-GAN Architecture with Tied Conv2D Trunk (Algorithm 2 box removed)."""
    fig, ax = plt.subplots(figsize=(12.2, 4.9), dpi=600)
    ax.axis('off')
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 46)

    c_dark = '#0F172A'
    c_blue = '#1D4ED8'
    c_gray = '#334155'
    c_rule = '#CBD5E1'

    def draw_node(x, y, w, h, title, lines, border=c_dark, bg='#FFFFFF',
                  title_color=c_dark, lw=1.5):
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.15,rounding_size=0.6",
            edgecolor=border, facecolor=bg, linewidth=lw
        )
        ax.add_patch(rect)

        header_y = y + h - 3.8
        ax.text(x + w / 2, header_y, title, ha='center', va='center',
                color=title_color, fontsize=8.4, fontweight='bold')

        rule_y = y + h - 6.4
        ax.plot([x + 1.2, x + w - 1.2], [rule_y, rule_y], color=c_rule, lw=0.9)

        body_y = y + (rule_y - y) / 2
        ax.text(x + w / 2, body_y, "\n".join(lines), ha='center', va='center',
                color='#0F172A', fontsize=7.2, multialignment='center', linespacing=1.35)

    def draw_arrow(x1, y1, x2, y2, label=""):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(facecolor=c_gray, edgecolor=c_gray,
                            arrowstyle='-|>', lw=1.4, mutation_scale=12)
        )
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 1.1, label,
                    ha='center', va='bottom', fontsize=7.3, fontweight='bold', color=c_dark)

    # Top Stream (Synthetic Path)
    top_y, row_h = 26.5, 16.5
    draw_node(1.5, top_y, 15.0, row_h, "Latent Prior",
              [r"$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_{100})$",
               "Gaussian Noise Vector"])
    draw_arrow(16.7, top_y + row_h / 2, 20.8, top_y + row_h / 2, r"$\mathbf{z}$")

    draw_node(21.0, top_y, 19.0, row_h, "Multi-Gen Bank",
              ["5 Dedicated Generators:",
               r"$G_1(\mathbf{z}) \dots G_5(\mathbf{z})$",
               r"MLP: $100 \to 128 \to 256 \to 78$"],
              border=c_blue, bg='#EFF6FF', title_color=c_blue, lw=1.8)
    draw_arrow(40.2, top_y + row_h / 2, 44.3, top_y + row_h / 2, r"$\hat{\mathbf{x}}_k$")

    draw_node(44.5, top_y, 15.5, row_h, "Synthetic Matrix",
              ["Spatial Reshape",
               r"$\tilde{\mathbf{x}} \in \mathbb{R}^{13 \times 6 \times 1}$"])

    # Bottom Stream (Real Data Path)
    bot_y = 3.0
    draw_node(1.5, bot_y, 15.0, row_h, "Real Flow Data",
              ["CICIDS2017 Dataset",
               "78 Numerical Features"])
    draw_arrow(16.7, bot_y + row_h / 2, 20.8, bot_y + row_h / 2, r"$\mathbf{x}_{\mathrm{raw}}$")

    draw_node(21.0, bot_y, 19.0, row_h, "Normalization",
              ["Zero-Leakage Scaler",
               r"$\tilde{x}_i = \frac{x_i - \min}{\max - \min}$",
               r"Bounded $\mathbf{x} \in [0, 1]^{78}$"])
    draw_arrow(40.2, bot_y + row_h / 2, 44.3, bot_y + row_h / 2, r"$\mathbf{x}_{\mathrm{norm}}$")

    draw_node(44.5, bot_y, 15.5, row_h, "Real Matrix",
              ["Spatial Reshape",
               r"$\mathbf{X} \in \mathbb{R}^{13 \times 6 \times 1}$"])

    # Merge Bus -> Tied Conv2D Trunk
    y_top_mid = top_y + row_h / 2
    y_bot_mid = bot_y + row_h / 2
    y_center = (y_top_mid + y_bot_mid) / 2
    bus_x = 63.0

    ax.plot([60.2, bus_x], [y_top_mid, y_top_mid], color=c_gray, lw=1.4)
    ax.plot([60.2, bus_x], [y_bot_mid, y_bot_mid], color=c_gray, lw=1.4)
    ax.plot([bus_x, bus_x], [y_bot_mid, y_top_mid], color=c_gray, lw=1.4)
    draw_arrow(bus_x, y_center, 67.3, y_center, r"$\mathbf{X}_{\mathrm{2D}}$")

    draw_node(67.5, 5.0, 18.0, 36.0, "Tied Conv2D Trunk",
              ["Shared Feature Extractor:",
               r"$3 \times 3$ Conv2D (32 Filters)",
               r"$+$ Spectral Norm $\mathbf{W}_{\mathrm{SN}}$",
               r"$+$ LeakyReLU($\alpha = 0.2$)",
               r"$+$ Dropout($p = 0.25$)",
               r"Flatten $\to$ Dense(256)",
               r"$+$ Spectral Normalization",
               r"Latent Features $\mathbf{h}$"],
              border=c_blue, bg='#EFF6FF', title_color=c_blue, lw=1.8)

    # Dual Output Heads
    draw_arrow(85.7, y_top_mid, 89.3, y_top_mid)
    draw_arrow(85.7, y_bot_mid, 89.3, y_bot_mid)

    draw_node(89.5, top_y, 17.0, row_h, r"Critic Head $D(\mathbf{x})$",
              ["Dense(1) Linear Output",
               r"Wasserstein Score $D(\mathbf{x}) \in \mathbb{R}$",
               "Real ($+$) vs Synthetic ($-$)"])

    draw_node(89.5, bot_y, 17.0, row_h, r"Classifier $C(\mathbf{x})$",
              ["Dense(6) $+$ Softmax",
               r"Probabilities $C(\mathbf{x}) \in [0, 1]^6$",
               r"Cross-Entropy Loss $\mathcal{L}_C$"])

    plt.tight_layout()
    out_path = PAPER_FIGS / "fig2_model_architecture.png"
    plt.savefig(out_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2 (600 DPI): {out_path}")


def draw_tsne_manifold(samples_per_class: int = 400):
    """
    Figure 6: High-resolution (600 DPI) 2D t-SNE projection with horizontal top legend,
    sized for crisp IEEE single-column legibility without retraining TMG-GAN.
    """
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import torch
    from sklearn.manifold import TSNE

    from src import config
    from src.dataset import CICIDSTrainingData
    from src.tmg_gan import TMGGAN

    gen_weights_path = ROOT / "models" / "tmg_gan_generators.pt"
    cd_weights_path = ROOT / "models" / "tmg_gan_cd.pt"
    data_dir = ROOT / "data" / "datasets" / "CICIDS2017"

    if not gen_weights_path.exists() or not data_dir.exists():
        print("Skipping Fig 6 regeneration: missing model weights or dataset.")
        return

    torch.manual_seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    dataset = CICIDSTrainingData(data_dir=data_dir, device=config.DEVICE)
    tmg_gan = TMGGAN(
        feature_num=config.FEATURE_NUM,
        label_num=config.LABEL_NUM,
        z_size=config.Z_SIZE,
        device=config.DEVICE,
    )
    gen_states = torch.load(gen_weights_path, map_location=config.DEVICE, weights_only=True)
    for g, state in zip(tmg_gan.generators, gen_states):
        g.load_state_dict(state)
    if cd_weights_path.exists():
        tmg_gan.cd.load_state_dict(
            torch.load(cd_weights_path, map_location=config.DEVICE, weights_only=True)
        )

    x_list, labels_list, origin_list = [], [], []
    for c in range(config.LABEL_NUM):
        class_name = config.CLASSES[c]
        real_all = dataset.samples_by_class[c]
        num_real = min(samples_per_class, len(real_all))
        idx = torch.randperm(len(real_all))[:num_real]
        x_list.append(real_all[idx].numpy())
        labels_list.extend([class_name] * num_real)
        origin_list.extend(["Real"] * num_real)

        gen_sub = tmg_gan.generate_raw_samples(c, num_real).numpy()
        x_list.append(gen_sub)
        labels_list.extend([class_name] * num_real)
        origin_list.extend(["TMG-GAN Generated"] * num_real)

    X = np.vstack(x_list)
    print(f"Running t-SNE on {len(X):,} flows for 600 DPI Fig 6...")
    tsne = TSNE(n_components=2, random_state=config.SEED, init="pca", learning_rate="auto")
    embedded = tsne.fit_transform(X)

    plot_df = pd.DataFrame({
        "Dim 1": embedded[:, 0],
        "Dim 2": embedded[:, 1],
        "Class": labels_list,
        "Type": origin_list,
    })

    palette = sns.color_palette("tab10", n_colors=config.LABEL_NUM)
    class_colors = dict(zip(config.CLASSES, palette))

    # Sized for IEEE column readability with crisp 600 DPI rasterization
    fig, ax = plt.subplots(figsize=(6.8, 5.4), dpi=600)

    # Plot Real (circles) and TMG-GAN Generated (X markers) cleanly without internal legend
    for origin, marker, size, alpha in [
        ("Real", "o", 18, 0.62),
        ("TMG-GAN Generated", "X", 22, 0.72),
    ]:
        sub = plot_df[plot_df["Type"] == origin]
        for cls in config.CLASSES:
            cls_sub = sub[sub["Class"] == cls]
            ax.scatter(
                cls_sub["Dim 1"],
                cls_sub["Dim 2"],
                c=[class_colors[cls]],
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors="white" if marker == "o" else "none",
                linewidths=0.35 if marker == "o" else 0.0,
            )

    ax.set_xlabel("Dim 1", fontsize=10.5, fontweight="bold")
    ax.set_ylabel("Dim 2", fontsize=10.5, fontweight="bold")
    ax.tick_params(axis="both", labelsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.28, linewidth=0.6)

    # Horizontal Legend Row 1: Traffic Classes (6 across)
    class_handles = [
        Line2D([0], [0], marker="o", color="w", label=cls,
               markerfacecolor=class_colors[cls], markersize=6.5)
        for cls in config.CLASSES
    ]
    # Horizontal Legend Row 2: Real vs Generated Marker Types (2 across)
    type_handles = [
        Line2D([0], [0], marker="o", color="w", label="Real",
               markerfacecolor="#475569", markersize=6.5),
        Line2D([0], [0], marker="X", color="w", label="TMG-GAN Generated",
               markerfacecolor="#475569", markersize=7.0),
    ]

    leg_classes = ax.legend(
        handles=class_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.075),
        ncol=6,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=8.2,
        handletextpad=0.3,
        columnspacing=0.9,
        borderpad=0.35,
    )
    ax.add_artist(leg_classes)

    ax.legend(
        handles=type_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.005),
        ncol=2,
        frameon=True,
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        fontsize=8.2,
        handletextpad=0.4,
        columnspacing=1.8,
        borderpad=0.3,
    )

    fig.suptitle(
        "t-SNE Projection: CICIDS2017 Real Flows vs TMG-GAN Generated Flows",
        fontsize=11.0,
        fontweight="bold",
        y=0.985,
    )

    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.11, right=0.97)

    raw_plot_path = ROOT / "data" / "plots" / "tsne_real_vs_generated.png"
    raw_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(raw_plot_path, dpi=600, bbox_inches="tight")

    out_path = PAPER_FIGS / "fig6_tsne_manifold.png"
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved Figure 6 (600 DPI): {out_path}")
    print(f"Updated raw t-SNE plot (600 DPI): {raw_plot_path}")


def copy_existing_plots():
    copies = [
        (ROOT / "results" / "benign_fpr.png", PAPER_FIGS / "fig3_benign_fpr.png"),
        (ROOT / "results" / "recall_by_class.png", PAPER_FIGS / "fig4_recall_by_class.png"),
        (ROOT / "results" / "confusion_tmg_gan.png", PAPER_FIGS / "fig5_confusion_matrix.png"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"Verified {dst.name}")


if __name__ == "__main__":
    draw_system_pipeline()
    draw_model_architecture()
    copy_existing_plots()
    draw_tsne_manifold(samples_per_class=400)
    print("All figures regenerated successfully!")
