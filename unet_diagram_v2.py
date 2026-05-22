"""
UNet Architecture Diagram Generator v2
========================================
True U-shape: encoder descends left, bottleneck at bottom centre,
decoder ascends right. Each residual block shows individual conv layers
and its internal skip connection with ⊕ merge.

Usage:
    from unet_diagram_v2 import draw_unet_diagram, MockResidualUNet
    m = MockResidualUNet(depth=3, num_res_blocks=2, layers_per_block=3)
    draw_unet_diagram(m, output_path="unet_v2")
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

C = dict(
    bg           = "#0D1117",
    panel        = "#161B22",
    border       = "#30363D",
    stem         = "#1F6FEB",
    stem_dark    = "#0D3B8A",
    enc          = "#388BFD",
    enc_dark     = "#1158C7",
    bot          = "#8957E5",
    bot_dark     = "#5A32A3",
    dec          = "#E05252",
    dec_dark     = "#9B2121",
    head         = "#3FB950",
    head_dark    = "#1A5C25",
    conv_bg      = "#21262D",
    conv_border  = "#484F58",
    conv_text    = "#C9D1D9",
    res_skip     = "#F0883E",
    unet_skip    = "#58A6FF",
    arrow        = "#8B949E",
    text_main    = "#F0F6FC",
    text_dim     = "#6E7681",
    text_label   = "#C9D1D9",
    pool_col     = "#56D364",
    interp_col   = "#F78166",
    ch_col       = "#D2A8FF",
)
MONO  = "DejaVu Sans Mono"
SANS  = "DejaVu Sans"


# ─── Primitives ───────────────────────────────────────────────────────────────

def _lerp(h1, h2, t):
    r = lambda h, s: int(h[s:s+2], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r(h1,1)+(r(h2,1)-r(h1,1))*t),
        int(r(h1,3)+(r(h2,3)-r(h1,3))*t),
        int(r(h1,5)+(r(h2,5)-r(h1,5))*t))

def rbox(ax, x, y, w, h, fc, ec="#00000000", lw=1.0, r=0.04, z=3, a=1.0):
    ax.add_patch(FancyBboxPatch((x,y), w, h, boxstyle=f"round,pad={r}",
                                fc=fc, ec=ec, lw=lw, zorder=z, alpha=a))

def txt(ax, x, y, s, fs=7, col=None, ha="center", va="center",
        bold=False, font=SANS, z=10):
    ax.text(x, y, s, fontsize=fs, color=col or C["text_main"],
            ha=ha, va=va, fontweight="bold" if bold else "normal",
            fontfamily=font, zorder=z)

def varrow(ax, x, y0, y1, col=None, lw=1.4, z=8):
    ax.annotate("", xy=(x,y1), xytext=(x,y0),
        arrowprops=dict(arrowstyle="->", color=col or C["arrow"],
                        lw=lw, connectionstyle="arc3,rad=0"), zorder=z)

def harrow(ax, x0, x1, y, col=None, lw=1.4, z=8):
    ax.annotate("", xy=(x1,y), xytext=(x0,y),
        arrowprops=dict(arrowstyle="->", color=col or C["arrow"],
                        lw=lw, connectionstyle="arc3,rad=0"), zorder=z)


# ─── Residual block ────────────────────────────────────────────────────────────

def res_block(ax, cx, y_top, bw, lpb, fc, dc, ch_in, ch_out, title):
    """
    Draw one residual block with its top edge at y_top.
    Returns (entry_y, exit_y) – signal enters at entry_y, exits at exit_y.
    Both are *inside* the block (close to top/bottom edges).
    """
    LH   = 0.42     # conv-layer row height
    HH   = 0.36     # header height
    FH   = 0.36     # footer (⊕) height
    PAD  = 0.10
    ILGP = 0.05     # inter-layer gap
    SW   = 0.22     # skip lane width

    total_h = PAD + HH + PAD*0.6 + lpb*(LH+ILGP) - ILGP + PAD + FH + PAD
    y_bot   = y_top - total_h

    # Outer frame
    rbox(ax, cx-bw/2, y_bot, bw, total_h, C["panel"], fc, lw=1.8, r=0.06, z=3)

    # Header
    hy = y_top - PAD - HH
    rbox(ax, cx-bw/2+0.06, hy, bw-0.12, HH, fc, dc, lw=1.0, r=0.03, z=4)
    txt(ax, cx, hy+HH*0.63, title, fs=6.8, bold=True, font=SANS)
    txt(ax, cx, hy+HH*0.17, f"{ch_in} → {ch_out} ch",
        fs=5.4, col=C["ch_col"], font=MONO)

    # Conv layer rows
    LX = cx - bw/2 + SW + 0.04
    LW = bw - SW - 0.08
    cursor = hy - PAD*0.6
    layer_tops, layer_bots = [], []

    for i in range(lpb):
        lt = cursor
        lb = cursor - LH
        layer_tops.append(lt)
        layer_bots.append(lb)
        rbox(ax, LX, lb+0.03, LW, LH-0.06, C["conv_bg"], C["conv_border"],
             lw=0.8, r=0.03, z=5)
        txt(ax, LX+LW/2, lb+LH*0.65, "Conv2d  3×3", fs=5.6, col=C["conv_text"], font=MONO)
        bn = "BN → ReLU" if i < lpb-1 else "BN"
        txt(ax, LX+LW/2, lb+LH*0.22, bn, fs=5.0, col=C["text_dim"], font=MONO)
        cursor = lb - ILGP
        if i < lpb-1:
            varrow(ax, cx, lb+0.03, lb-ILGP+LH-0.03, C["conv_text"], lw=0.9, z=7)

    # ⊕ merge circle
    merge_y = cursor - PAD*0.6 - FH/2
    mr = 0.115
    ax.add_patch(plt.Circle((cx, merge_y), mr,
                 fc=C["panel"], ec=C["res_skip"], lw=1.6, zorder=7))
    txt(ax, cx, merge_y, "⊕", fs=9.5, col=C["res_skip"], font=SANS)

    # Arrow: last conv layer → ⊕
    varrow(ax, cx, layer_bots[-1]+0.03, merge_y+mr+0.02, C["conv_text"], lw=0.9, z=7)

    # Residual skip (left lane)
    sx   = cx - bw/2 + SW*0.42
    s_t  = layer_tops[0] - 0.02
    ax.plot([LX-0.03, sx], [s_t, s_t], color=C["res_skip"], lw=1.6, zorder=6)
    ax.plot([sx, sx], [s_t, merge_y+0.02], color=C["res_skip"], lw=1.6, zorder=6)
    ax.annotate("", xy=(cx-mr-0.02, merge_y), xytext=(sx, merge_y),
        arrowprops=dict(arrowstyle="->", color=C["res_skip"], lw=1.6,
                        connectionstyle="arc3,rad=0"), zorder=8)

    # 1×1 projection badge if ch_in ≠ ch_out
    if ch_in != ch_out:
        mid_s = (s_t + merge_y) / 2
        rbox(ax, sx-0.115, mid_s-0.115, 0.23, 0.23,
             "#2D1E0F", C["res_skip"], lw=0.9, r=0.03, z=9)
        txt(ax, sx, mid_s, "1×1", fs=5.2, col=C["res_skip"], font=MONO, z=10)

    # Entry / exit arrows
    entry_y = y_top - PAD*0.35
    exit_y  = y_bot + PAD*0.35
    varrow(ax, cx, entry_y, layer_tops[0]-0.03, C["conv_text"], lw=1.0, z=7)
    varrow(ax, cx, merge_y-mr-0.02, exit_y, C["conv_text"], lw=1.0, z=7)

    return entry_y, exit_y, y_top, y_bot


# ─── Tag pill (pool / upsample) ────────────────────────────────────────────────

def tag(ax, cx, cy, label, col):
    w = len(label) * 0.068 + 0.20
    rbox(ax, cx-w/2, cy-0.115, w, 0.23, "#0D0D0D", col, lw=1.0, r=0.04, z=7)
    txt(ax, cx, cy, label, fs=5.6, col=col, font=MONO, z=8)


# ─── Main ─────────────────────────────────────────────────────────────────────

def draw_unet_diagram(model, output_path="unet_diagram_v2",
                      fmt=("svg", "png"), dpi=180):
    """
    Draw a U-shaped Residual U-Net with expanded residual block detail.
    Encoder descends the left; decoder ascends the right;
    bottleneck sits at the bottom centre.
    """

    # ── Introspect ────────────────────────────────────────────────────────────
    depth   = len(model.encoders)
    nrb     = len(model.bottleneck)
    lpb     = sum(1 for m in model.encoders[0].res_blocks[0].block
                  if hasattr(m, "out_channels"))
    enc_ch  = [enc.res_blocks[0].block[0].out_channels for enc in model.encoders]
    bot_ch  = model.bottleneck[0].block[0].out_channels
    out_ch  = model.head.out_channels

    # ── Geometry ──────────────────────────────────────────────────────────────
    LH, HH, FH = 0.42, 0.36, 0.36
    PAD, ILGP  = 0.10, 0.05
    RB_H = PAD + HH + PAD*0.6 + lpb*(LH+ILGP) - ILGP + PAD + FH + PAD
    RB_W = 2.35

    RB_GAP  = 0.30    # gap between stacked res-blocks within one level
    LVL_GAP = 0.60    # gap between encoder/decoder levels

    lvl_h   = nrb * RB_H + (nrb-1) * RB_GAP

    # Horizontal
    SKIP_LANE = 1.90
    MARGIN    = 1.20
    ENC_CX    = MARGIN + RB_W/2
    DEC_CX    = MARGIN + RB_W + SKIP_LANE + RB_W/2
    BOT_CX    = (ENC_CX + DEC_CX) / 2
    FIG_W     = 2*MARGIN + 2*RB_W + SKIP_LANE + 0.3

    # Vertical – level 0 at top, descending
    STEM_H  = 0.38
    T_PAD   = 1.50   # above level-0 (stem + input label)
    B_PAD   = 1.80   # below bottleneck (legend)

    total_enc_h = depth * lvl_h + (depth-1) * LVL_GAP
    # Top of level i (from bottom of figure):
    #   y_top_lvl[0] = FIG_H - T_PAD
    #   y_top_lvl[i] = y_top_lvl[0] - i*(lvl_h + LVL_GAP)

    # We need to compute FIG_H from the content:
    #   T_PAD  + total_enc_h  + LVL_GAP (enc-to-bot transition)
    #   + bot_h + B_PAD
    BOT_H = nrb * RB_H + (nrb-1) * RB_GAP   # horizontal, but just for height
    FIG_H = T_PAD + total_enc_h + LVL_GAP + BOT_H + B_PAD

    def lvl_top_y(i):
        return FIG_H - T_PAD - i * (lvl_h + LVL_GAP)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_facecolor(C["bg"]); fig.patch.set_facecolor(C["bg"])
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(0, FIG_W); ax.set_ylim(0, FIG_H)

    # ── Encoder column ────────────────────────────────────────────────────────
    enc_lvl_first_entry = []   # entry_y of block-0 of each level
    enc_lvl_last_exit   = []   # exit_y  of last block of each level

    for lvl in range(depth):
        lt = lvl_top_y(lvl)
        prev_exit   = None
        first_entry = None
        last_exit   = None

        for rb in range(nrb):
            rb_top = lt - rb * (RB_H + RB_GAP)
            ci = enc_ch[lvl-1] if (lvl > 0 and rb == 0) else enc_ch[lvl]
            co = enc_ch[lvl]
            t  = lvl / max(depth-1, 1)
            fc = _lerp(C["enc"], C["enc_dark"], t*0.55)
            dc = _lerp(C["enc_dark"], C["bot_dark"], t*0.55)
            en, ex, _, _ = res_block(ax, ENC_CX, rb_top, RB_W, lpb, fc, dc,
                                     ci, co, f"Enc {lvl+1}  ·  Block {rb+1}")
            if rb == 0:        first_entry = en
            if rb == nrb - 1:  last_exit   = ex
            if prev_exit is not None:
                varrow(ax, ENC_CX, prev_exit, en)
            prev_exit = ex

        enc_lvl_first_entry.append(first_entry)
        enc_lvl_last_exit.append(last_exit)

        # Pool tag + arrow to next level
        if lvl < depth - 1:
            pool_cy = last_exit - 0.25
            tag(ax, ENC_CX, pool_cy, "MaxPool 2×2", C["pool_col"])
            varrow(ax, ENC_CX, pool_cy - 0.115, lvl_top_y(lvl+1) + 0.02)

    # Stem bar
    stem_bot = enc_lvl_first_entry[0] + 0.08
    rbox(ax, ENC_CX-RB_W/2+0.12, stem_bot, RB_W-0.24, STEM_H,
         C["stem"], C["stem_dark"], lw=1.5, r=0.04, z=5)
    txt(ax, ENC_CX, stem_bot+STEM_H/2, f"Stem  ·  {enc_ch[0]}ch",
        fs=7.2, bold=True, font=SANS)
    varrow(ax, ENC_CX, stem_bot+STEM_H, enc_lvl_first_entry[0])
    txt(ax, ENC_CX, stem_bot+STEM_H+0.38,
        "▼  INPUT", fs=7.5, bold=True, col=C["text_dim"], font=SANS)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    # Sits below the deepest encoder level
    bot_top_y  = enc_lvl_last_exit[-1] - 0.60
    bot_total_w = nrb*RB_W + (nrb-1)*RB_GAP
    bot0_cx    = BOT_CX - bot_total_w/2 + RB_W/2
    bot_entries = []
    prev_b_exit = None
    prev_b_cx   = None

    for rb in range(nrb):
        bx  = bot0_cx + rb*(RB_W + RB_GAP)
        ci  = enc_ch[-1] if rb == 0 else bot_ch
        co  = bot_ch
        en, ex, _, boty = res_block(ax, bx, bot_top_y, RB_W, lpb,
                                    C["bot"], C["bot_dark"], ci, co,
                                    f"Bottleneck · Block {rb+1}")
        bot_entries.append((bx, en, ex, boty))
        if prev_b_exit is not None:
            harrow(ax, prev_b_cx+RB_W/2+0.04, bx-RB_W/2-0.04,
                   bot_top_y - RB_H/2)
        prev_b_exit = ex
        prev_b_cx   = bx

    # Elbow: enc deepest exit → bot block-0 entry
    ax.annotate("", xy=(bot0_cx, bot_top_y+0.06),
                xytext=(ENC_CX, enc_lvl_last_exit[-1]-0.05),
                arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.8,
                                connectionstyle="angle,angleA=270,angleB=180,rad=0.18"),
                zorder=8)
    txt(ax, BOT_CX, bot_entries[0][3]-0.28,
        f"Bottleneck  ·  {bot_ch}ch", fs=7.5, bold=True, col=C["bot"], font=SANS)

    # ── Decoder column ────────────────────────────────────────────────────────
    # Decoder level 0 = deepest (mirrors encoder level depth-1),
    # Decoder level (depth-1) = shallowest (mirrors encoder level 0).
    # Visually: dec lvl 0 is at the BOTTOM right, dec lvl (depth-1) at the TOP right.
    # We iterate d_lvl = 0..depth-1; enc_lvl = depth-1-d_lvl
    # Vertical top for decoder level d_lvl = same as encoder level enc_lvl

    dec_lvl_first_entry = []
    dec_lvl_last_exit   = []

    for d_lvl in range(depth):
        enc_lvl  = depth - 1 - d_lvl
        lt       = lvl_top_y(enc_lvl)
        prev_exit   = None
        first_entry = None
        last_exit   = None

        for rb in range(nrb):
            rb_top = lt - rb * (RB_H + RB_GAP)
            ci = enc_ch[enc_lvl] * 2 if rb == 0 else enc_ch[enc_lvl]
            co = enc_ch[enc_lvl]
            t  = d_lvl / max(depth-1, 1)
            fc = _lerp(C["dec_dark"], C["dec"], t)
            dc = _lerp(C["bot_dark"], C["dec_dark"], t)
            en, ex, _, _ = res_block(ax, DEC_CX, rb_top, RB_W, lpb, fc, dc,
                                     ci, co, f"Dec {d_lvl+1}  ·  Block {rb+1}")
            if rb == 0:        first_entry = en
            if rb == nrb - 1:  last_exit   = ex
            if prev_exit is not None:
                varrow(ax, DEC_CX, prev_exit, en)
            prev_exit = ex

        dec_lvl_first_entry.append(first_entry)
        dec_lvl_last_exit.append(last_exit)

        # Upsample tag + arrow to next (shallower) dec level
        if d_lvl < depth - 1:
            up_cy = last_exit - 0.25
            tag(ax, DEC_CX, up_cy, "Bilinear ×2", C["interp_col"])
            # next dec level is d_lvl+1, which maps to enc_lvl-1
            next_enc_lvl = enc_lvl - 1
            varrow(ax, DEC_CX, up_cy - 0.115, lvl_top_y(next_enc_lvl) + 0.02)

    # Elbow: last bottleneck block exit → dec level-0 (deepest) entry
    last_bot = bot_entries[-1]
    ax.annotate("", xy=(DEC_CX, dec_lvl_first_entry[0] + 0.06),
                xytext=(last_bot[0], last_bot[2]-0.05),
                arrowprops=dict(arrowstyle="->", color=C["arrow"], lw=1.8,
                                connectionstyle="angle,angleA=270,angleB=0,rad=0.18"),
                zorder=8)

    # Output head — sits ABOVE the shallowest decoder level (mirrors enc stem)
    # dec_lvl_first_entry[-1] = entry_y of block-0 of d_lvl=depth-1 (top of right col)
    head_bot = dec_lvl_first_entry[-1] + 0.08
    rbox(ax, DEC_CX-RB_W/2+0.12, head_bot, RB_W-0.24, STEM_H,
         C["head"], C["head_dark"], lw=1.5, r=0.04, z=5)
    txt(ax, DEC_CX, head_bot+STEM_H/2,
        f"Output Head  ·  1×1 Conv  →  {out_ch}ch",
        fs=7.0, bold=True, font=SANS)
    # Arrow: head bottom → into the first block of the shallowest dec level
    varrow(ax, DEC_CX, head_bot, dec_lvl_first_entry[-1], C["arrow"])
    txt(ax, DEC_CX, head_bot+STEM_H+0.38,
        "▲  OUTPUT", fs=7.5, bold=True, col=C["text_dim"], font=SANS)

    # ── U-Net skip connections ────────────────────────────────────────────────
    # Enc level i ↔ Dec level (depth-1-i)  (both at the same vertical range)
    for enc_lvl in range(depth):
        mid_y = lvl_top_y(enc_lvl) - lvl_h / 2
        x0 = ENC_CX + RB_W/2 + 0.08
        x1 = DEC_CX - RB_W/2 - 0.08
        ax.plot([x0, x1], [mid_y, mid_y],
                color=C["unet_skip"], lw=2.0,
                linestyle=(0,(5,2.5)), zorder=7, alpha=0.9)
        ax.annotate("", xy=(x1-0.01, mid_y), xytext=(x1-0.32, mid_y),
                    arrowprops=dict(arrowstyle="-|>", color=C["unet_skip"],
                                    lw=2.0, mutation_scale=10), zorder=9)
        mx = (x0 + x1) / 2
        w  = 1.12
        rbox(ax, mx-w/2, mid_y-0.12, w, 0.24, "#0D1B2E", C["unet_skip"],
             lw=0.9, r=0.03, z=10)
        txt(ax, mx, mid_y, f"concat  ·  {enc_ch[enc_lvl]}ch",
            fs=5.8, col=C["unet_skip"], font=MONO, z=11)

    # ── Column headers ────────────────────────────────────────────────────────
    hy = FIG_H - 0.58
    for cx, label, col in [
        (ENC_CX, "ENCODER",    C["enc"]),
        (BOT_CX, "BOTTLENECK", C["bot"]),
        (DEC_CX, "DECODER",    C["dec"]),
    ]:
        txt(ax, cx, hy, label, fs=9.5, bold=True, col=col, font=SANS)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.set_title(
        f"Residual U-Net  ·  depth={depth}  ·  "
        f"{nrb} res-block(s)/level  ·  {lpb} conv layers/block  ·  "
        f"bottleneck {bot_ch}ch",
        fontsize=9.5, fontweight="bold",
        color=C["text_main"], fontfamily=SANS, pad=16)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mpatches.Patch(fc=C["enc"],    ec=C["enc_dark"], lw=0.8, label="Encoder block"),
        mpatches.Patch(fc=C["bot"],    ec=C["bot_dark"], lw=0.8, label="Bottleneck block"),
        mpatches.Patch(fc=C["dec"],    ec=C["dec_dark"], lw=0.8, label="Decoder block"),
        mpatches.Patch(fc=C["conv_bg"],ec=C["conv_border"], lw=0.8, label="Conv2d → BN → ReLU"),
        Line2D([0],[0], color=C["res_skip"],  lw=2.0, label="Residual skip  ⊕"),
        Line2D([0],[0], color=C["unet_skip"], lw=2.0, ls=(0,(5,2)), label="U-Net skip (concat)"),
        mpatches.Patch(fc=C["bg"], ec=C["pool_col"],   lw=1.2, label="MaxPool 2×2"),
        mpatches.Patch(fc=C["bg"], ec=C["interp_col"], lw=1.2, label="Bilinear upsample"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4,
              bbox_to_anchor=(0.5, 0.003),
              framealpha=0.97, fontsize=7.2,
              facecolor=C["panel"], edgecolor=C["border"],
              prop={"family": SANS, "size": 7.2},
              labelcolor=C["text_label"])

    plt.tight_layout(pad=0.5)
    saved = []
    for f in fmt:
        path = f"{output_path}.{f}"
        fig.savefig(path, format=f, dpi=dpi,
                    bbox_inches="tight", facecolor=C["bg"])
        saved.append(path)
        print(f"Saved: {path}")
    plt.close(fig)
    return saved


# ── Mock ──────────────────────────────────────────────────────────────────────

class _C:
    def __init__(self, o): self.out_channels = o
class _RB:
    def __init__(self, o, l): self.block = [_C(o) for _ in range(l)]
class _EB:
    def __init__(self, o, n, l): self.res_blocks = [_RB(o, l) for _ in range(n)]
class _H:
    def __init__(self, o): self.out_channels = o

class MockResidualUNet:
    """Pure-Python stand-in for ResidualUNet — no PyTorch needed."""
    def __init__(self, in_channels=3, out_channels=1, base_channels=32,
                 depth=4, num_res_blocks=2, layers_per_block=3, channel_mult=2.0):
        ec = [int(base_channels * channel_mult**i) for i in range(depth)]
        bc = int(ec[-1] * channel_mult)
        self.encoders   = [_EB(ec[i], num_res_blocks, layers_per_block) for i in range(depth)]
        self.bottleneck = [_RB(bc, layers_per_block) for _ in range(num_res_blocks)]
        self.head       = _H(out_channels)


if __name__ == "__main__":
    configs = [
        dict(depth=3, num_res_blocks=2, layers_per_block=3,
             base_channels=32, out_channels=1),
        dict(depth=4, num_res_blocks=1, layers_per_block=2,
             base_channels=64, out_channels=3),
    ]
    for i, cfg in enumerate(configs):
        m = MockResidualUNet(**cfg)
        draw_unet_diagram(m,
                          output_path=f"/mnt/user-data/outputs/unet_diagram_v2_{i+1}",
                          fmt=("png", "svg"))
