"""
Generate publication-quality, perfectly aligned IEEE paper figures:
Figure 1: Pipeline Flowchart - pristine spacing, sleek directional flow, zero clutter.
Figure 2: Architecture & Algorithm 2 Filtering Gate - continuous bus routing and direct connections into the Quality Gate.
"""

import sys
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
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
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 1: {out_path}")


def draw_model_architecture():
    """Figure 2: Comprehensive TMG-GAN Architecture with Tied Conv2D Backbone and Algorithm 2 Filtering."""
    fig, ax = plt.subplots(figsize=(12, 6.4), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 64)

    c_gen = '#1E40AF'    # Blue
    c_real = '#334155'   # Slate
    c_trunk = '#C2410C'  # Amber/Orange
    c_head = '#0E7490'   # Cyan
    c_gate = '#7E22CE'   # Purple
    c_gray = '#475569'

    def draw_card(x, y, w, h, title, lines, color, bg='#FFFFFF', banner_h=4.8):
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=0.8",
                              edgecolor=color, facecolor=bg, linewidth=1.5)
        ax.add_patch(rect)
        if title:
            header = FancyBboxPatch((x, y + h - banner_h), w, banner_h,
                                    boxstyle="round,pad=0.1,rounding_size=0.4",
                                    edgecolor=color, facecolor=color, linewidth=0)
            ax.add_patch(header)
            ax.text(x + w/2, y + h - banner_h/2, title, ha='center', va='center',
                    color='white', fontsize=7.8, fontweight='bold')
            body_y = y + (h - banner_h)/2
        else:
            body_y = y + h/2
            
        ax.text(x + w/2, body_y, "\n".join(lines), ha='center', va='center',
                color='#0F172A', fontsize=7.1, multialignment='center', linespacing=1.3)

    # Main Diagram Title
    ax.text(52.5, 61.5, "TMG-GAN Neural Architecture & Sample Quality Verification Pipeline",
            ha='center', va='center', fontsize=11.5, fontweight='bold', color='#0F172A')

    # 1. TOP STREAM: Synthetic Generation
    draw_card(2.5, 39, 12, 18, "Latent Prior", 
              [r"$\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$", "Dim = 100", "Gaussian Prior"], 
              c_gen, '#EFF6FF')

    # Arrow -> Multi-Generator
    ax.annotate('', xy=(18.5, 48.0), xytext=(14.7, 48.0),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, arrowstyle='-|>', lw=1.5, mutation_scale=12))

    # Multi-Generator Bank
    draw_card(18.5, 35, 19, 23, "Multi-Generator Bank",
              [r"Class 1: $G_1(\mathbf{z}) \to \tilde{\mathbf{x}}_1$",
               r"Class 2: $G_2(\mathbf{z}) \to \tilde{\mathbf{x}}_2$",
               r"$\vdots$",
               r"Class $K$: $G_K(\mathbf{z}) \to \tilde{\mathbf{x}}_K$",
               "Architecture per Gen:",
               "Dense 100->128->256->78",
               "LeakyReLU + Min-Max Out"],
              c_gen, '#F8FAFC')

    # Arrow -> Synthetic Reshape
    ax.annotate('', xy=(41.5, 48.0), xytext=(37.7, 48.0),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, arrowstyle='-|>', lw=1.5, mutation_scale=12))

    # Synthetic Reshape
    draw_card(41.5, 39, 15, 18, "Synthetic Matrix",
              [r"$\tilde{\mathbf{x}}_k \in [0, 1]^{78}$",
               r"$\downarrow$",
               "Reshape to 2D",
               r"$\tilde{\mathbf{X}} \in \mathbb{R}^{13 \times 6 \times 1}$"],
              c_gen, '#EFF6FF')

    # 2. BOTTOM STREAM: Real Flow Records
    draw_card(18.5, 12, 19, 18, "Real Flow Training Set",
              ["Empirical Traffic",
               r"$(\mathbf{x}, y) \sim P_{\text{data}}(\mathbf{x}, y)$",
               "78 Flow Features",
               "Min-Max Scaled to [0, 1]"],
              c_real, '#F8FAFC')

    ax.annotate('', xy=(41.5, 21.0), xytext=(37.7, 21.0),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, arrowstyle='-|>', lw=1.5, mutation_scale=12))

    draw_card(41.5, 12, 15, 18, "Real Matrix",
              [r"$\mathbf{x}_{\text{real}} \in [0, 1]^{78}$",
               r"$\downarrow$",
               "Reshape to 2D",
               r"$\mathbf{X} \in \mathbb{R}^{13 \times 6 \times 1}$"],
              c_real, '#F8FAFC')

    # 3. TIED CONV2D BACKBONE with Continuous Bus Junction
    # Synthetic arm into vertical bus at x=59.5
    ax.plot([56.5, 59.5], [48.0, 48.0], color=c_gray, lw=1.4)
    # Real arm into vertical bus at x=59.5
    ax.plot([56.5, 59.5], [21.0, 21.0], color=c_gray, lw=1.4)
    # Solid vertical bus connecting both arms
    ax.plot([59.5, 59.5], [21.0, 48.0], color=c_gray, lw=1.4)
    # Single unified horizontal arrow into Tied Backbone
    ax.annotate('', xy=(62.5, 34.5), xytext=(59.5, 34.5),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, arrowstyle='-|>', lw=1.5, mutation_scale=12))

    draw_card(62.5, 15.5, 18.5, 36, "Tied Backbone Network",
              ["Input Matrix: " + r"$\mathbb{R}^{13 \times 6 \times 1}$",
               r"$\downarrow$",
               "Conv2D (32 filters, 3x3)",
               "+ Spectral Normalization",
               "+ LeakyReLU(0.2)",
               "+ Dropout(0.25)",
               r"$\downarrow$",
               "Flatten -> Dense(256)",
               "+ Spectral Normalization",
               r"$\downarrow$",
               "Shared Latent Features"],
              c_trunk, '#FFFBEB', banner_h=5.0)

    # 4. DUAL OUTPUT HEADS with Continuous Bus Junction
    # Right exit from backbone to vertical distribution bus at x=83.5
    ax.plot([81.0, 83.5], [34.5, 34.5], color=c_gray, lw=1.4)
    # Vertical distribution bus
    ax.plot([83.5, 83.5], [22.0, 46.0], color=c_gray, lw=1.4)
    # Branch into Critic Head
    ax.annotate('', xy=(85.5, 46.0), xytext=(83.5, 46.0),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, arrowstyle='-|>', lw=1.5, mutation_scale=12))
    # Branch into Auxiliary Head
    ax.annotate('', xy=(85.5, 22.0), xytext=(83.5, 22.0),
                arrowprops=dict(facecolor=c_gray, edgecolor=c_gray, arrowstyle='-|>', lw=1.5, mutation_scale=12))

    draw_card(85.5, 38, 17, 16, "Critic Head D(x)",
              ["Dense(1) Output",
               r"$D(\mathbf{x}) \in \mathbb{R}$",
               "Wasserstein Realism Score",
               "Real (+) vs Synthetic (-)"],
              c_head, '#ECFEFF', banner_h=4.5)

    draw_card(85.5, 14, 17, 16, "Auxiliary Head C(x)",
              ["Dense(6) + Softmax",
               r"$C(\mathbf{x}) \in [0, 1]^6$",
               "Cross-Entropy Loss",
               "Attack Class Probability"],
              c_head, '#ECFEFF', banner_h=4.5)

    # 5. ALGORITHM 2 QUALITY FILTER GATE (Integrated bottom section with feedback arrows)
    # Downward connector from Heads into Quality Gate
    ax.plot([94.0, 94.0], [14.0, 9.8], color=c_gate, lw=1.2, linestyle='--')
    ax.annotate('', xy=(94.0, 8.8), xytext=(94.0, 9.8),
                arrowprops=dict(facecolor=c_gate, edgecolor=c_gate, arrowstyle='-|>', lw=1.2, mutation_scale=10))

    rect_gate = FancyBboxPatch((2.5, 1.4), 100, 7.2, boxstyle="round,pad=0.2,rounding_size=0.6",
                               edgecolor=c_gate, facecolor='#FAF5FF', linewidth=1.4)
    ax.add_patch(rect_gate)

    ax.text(12.0, 5.0, "Algorithm 2 Gate:", fontsize=8.0, fontweight='bold', color=c_gate, ha='center', va='center')
    ax.text(52.0, 5.0,
            r"Candidate $\tilde{\mathbf{x}} = G_k(\mathbf{z})$ accepted iff:  " +
            r"$C(\tilde{\mathbf{x}})_k \geq 0.50$ (Class Confidence)  AND  " +
            r"$D(\tilde{\mathbf{x}}) \geq \tau_D$ (Realism Check)",
            fontsize=7.3, color='#3B0764', ha='center', va='center')
    ax.text(94.0, 5.0, r"Pass Rate: 98.10%", fontsize=7.8, fontweight='bold', color='#047857', ha='center', va='center')

    plt.tight_layout()
    out_path = PAPER_FIGS / "fig2_model_architecture.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Figure 2: {out_path}")

def copy_existing_plots():
    copies = [
        (ROOT / "results" / "benign_fpr.png", PAPER_FIGS / "fig3_benign_fpr.png"),
        (ROOT / "results" / "recall_by_class.png", PAPER_FIGS / "fig4_recall_by_class.png"),
        (ROOT / "results" / "confusion_tmg_gan.png", PAPER_FIGS / "fig5_confusion_matrix.png"),
        (ROOT / "data" / "plots" / "tsne_real_vs_generated.png", PAPER_FIGS / "fig6_tsne_manifold.png"),
    ]
    for src, dst in copies:
        if src.exists():
            shutil.copyfile(src, dst)
            print(f"Verified {dst.name}")

if __name__ == "__main__":
    draw_system_pipeline()
    draw_model_architecture()
    copy_existing_plots()
    print("All figures regenerated successfully!")
