import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None
    print("[CANH BAO] Chua cai 'scikit-image' -> pip install scikit-image. "
          "Neu khong co, script se fallback ve median-toan-mask (SAI, khong khop pipeline that).")

try:
    from scipy.ndimage import distance_transform_edt
except ImportError:
    distance_transform_edt = None

# Mau preview production dung (BGR) - phai KHOP voi _PREVIEW_COLOR_FILL/_SATIN
# trong opencv_stitch_classifier*.py
_COLOR_FILL_BGR = (0, 255, 255)    # Vang
_COLOR_SATIN_BGR = (255, 0, 255)   # Hong


def split_mixed_width_shape(shape_mask: np.ndarray, dist: np.ndarray, skel: np.ndarray,
                             pixel_to_mm: float, threshold_mm: float):
    """Xem giai thich chi tiet trong ban v4. Tach 1 shape co do rong bien
    thien manh (vd khoi rong + vien mong lien thong) thanh 2 vung con bang
    cach lan truyen nhan tu skeleton ra toan bo mask (nearest-neighbor)."""
    if distance_transform_edt is None or skel.sum() == 0:
        return None, None

    skel_thickness_mm = dist[skel] * 2.0 * pixel_to_mm
    skel_is_fill = skel_thickness_mm > threshold_mm

    skel_label_map = np.zeros(shape_mask.shape, dtype=np.uint8)
    ys, xs = np.where(skel)
    skel_label_map[ys, xs] = np.where(skel_is_fill, 1, 2)

    _, indices = distance_transform_edt(~skel, return_indices=True)
    nearest_label = skel_label_map[indices[0], indices[1]]

    final_label = np.zeros(shape_mask.shape, dtype=np.uint8)
    final_label[shape_mask] = nearest_label[shape_mask]

    k = np.ones((7, 7), np.uint8)
    fill_area = cv2.morphologyEx((final_label == 1).astype(np.uint8), cv2.MORPH_CLOSE, k)
    fill_area = cv2.morphologyEx(fill_area, cv2.MORPH_OPEN, k).astype(bool)
    satin_area = shape_mask & (~fill_area)

    return fill_area, satin_area


def load_image_masks(image_path: str, color_tolerance: int = 25):
    """
    [SUA LOI QUAN TRONG] Ham nay THAY THE hoan toan cach lam cu
    (cvtColor -> grayscale -> Otsu). Cach cu CHỈ tach duoc 2 cum theo DO
    SANG (sang/toi), nen voi anh preview co 2 MAU MUC khac nhau (FILL=vang
    sang, SATIN=hong toi hon nhieu ve grayscale), Otsu se de dang gop
    nham 1 trong 2 mau mucVAO CHUNG voi nen den - lam bien mat toan bo
    vung co mau do khoi phep do (chinh la loi ban gap voi dau "+" mau hong
    trong 155_pred.png: Otsu chon threshold = 105, TRUNG KHOP voi gia tri
    grayscale cua mau hong (255,0,255) -> ca vung hong bi tinh la background).

    Ham moi nay THU nhan dien 2 mau chuan cua production preview (vang=FILL,
    hong=SATIN) TRUOC bang cach so mau BGR truc tiep (khong qua grayscale),
    tach thanh 2 mask RIENG BIET. Neu anh KHONG phai kieu preview 2-mau nay
    (vd anh nguon don sac trang/den nhu test luc dau) thi TU DONG fallback
    ve cach cu (grayscale+Otsu) de van dung duoc voi moi loai anh.

    Tra ve:
      mode: 'classified' (anh co ca FILL+SATIN, tach rieng duoc) hoac
            'generic' (anh 1 mau, dung fallback Otsu)
      fill_mask, satin_mask: np.ndarray bool, rieng cho tung loai (rong het
            neu mode='generic', luc do dung total_fg thay the)
      total_fg: np.ndarray bool - hop cua ca 2 (hoac toan bo foreground neu generic)
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Khong tim thay anh: {image_path}")

    h, w = img.shape[:2]

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[:, :, :3]
        alpha = img[:, :, 3]
    else:
        bgr = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        alpha = None

    def _color_mask(bgr_img, target_bgr, tol):
        b, g, r = target_bgr
        return (np.abs(bgr_img[:, :, 0].astype(int) - b) <= tol) & \
               (np.abs(bgr_img[:, :, 1].astype(int) - g) <= tol) & \
               (np.abs(bgr_img[:, :, 2].astype(int) - r) <= tol)

    fill_mask = _color_mask(bgr, _COLOR_FILL_BGR, color_tolerance)
    satin_mask = _color_mask(bgr, _COLOR_SATIN_BGR, color_tolerance)
    n_fill, n_satin = int(fill_mask.sum()), int(satin_mask.sum())

    # Nguong: can it nhat ~0.05% dien tich anh cho MOI mau thi moi coi la
    # "anh preview 2-mau chuan", tranh nham vai pixel trung mau ngau nhien
    # trong anh nguon thanh tin hieu gia.
    min_px = 0.0005 * h * w
    if n_fill >= min_px and n_satin >= min_px:
        total_fg = fill_mask | satin_mask
        return "classified", fill_mask, satin_mask, total_fg

    # --- Fallback: anh khong phai kieu 2-mau preview -> dung cach cu ---
    if alpha is not None:
        _, binary = cv2.threshold(alpha, 127, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        corners_sum = int(binary[0, 0]) + int(binary[0, -1]) + int(binary[-1, 0]) + int(binary[-1, -1])
        if corners_sum > 255 * 2:
            binary = cv2.bitwise_not(binary)
    total_fg = binary > 0
    empty = np.zeros((h, w), dtype=bool)
    return "generic", empty, empty, total_fg


def _measure_and_classify_shape(contour, holes, canvas_shape, pixel_to_mm, dynamic_threshold_mm,
                                 total_fg_area_px, ink_ratio_gate, solidity_gate,
                                 hysteresis_band, solidity_trust_minrect):
    h, w = canvas_shape
    shape_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(shape_mask, [contour], -1, 255, thickness=cv2.FILLED)
    for hole in holes:
        cv2.drawContours(shape_mask, [hole], -1, 0, thickness=cv2.FILLED)
    is_hollow = len(holes) > 0
    mask_bool = shape_mask > 0

    dist = cv2.distanceTransform(shape_mask, cv2.DIST_L2, 5)
    max_dist_px = float(np.max(dist))
    if max_dist_px == 0:
        return None

    max_thickness_mm = (max_dist_px * 2.0) * pixel_to_mm

    median_thickness_mm = max_thickness_mm
    n_skel_px = 0
    skel = None
    if skeletonize is not None:
        skel = skeletonize(mask_bool)
        skel_dist = dist[skel]
        n_skel_px = len(skel_dist)
        if n_skel_px > 0:
            median_thickness_mm = (np.median(skel_dist) * 2.0) * pixel_to_mm
    else:
        non_zero_dist = dist[dist > 0]
        if len(non_zero_dist) > 0:
            median_thickness_mm = (np.median(non_zero_dist) * 2.0) * pixel_to_mm

    true_area_px = float(np.count_nonzero(shape_mask))
    hull = cv2.convexHull(contour)
    hull_area_px = cv2.contourArea(hull)
    solidity = true_area_px / hull_area_px if hull_area_px > 0 else 1.0
    ink_ratio = true_area_px / total_fg_area_px if total_fg_area_px > 0 else 1.0

    skeleton_median_mm = median_thickness_mm

    rect = cv2.minAreaRect(contour)
    short_side_px, long_side_px = sorted(rect[1])
    minrect_short_mm = short_side_px * pixel_to_mm

    used_minrect = False
    thickness_source = "skeleton-median"
    if solidity >= solidity_trust_minrect and not is_hollow:
        if 0.5 * max_thickness_mm <= minrect_short_mm <= 1.2 * max_thickness_mm:
            median_thickness_mm = minrect_short_mm
            used_minrect = True
            thickness_source = "minAreaRect"

    thickness_mm = median_thickness_mm
    ratio = thickness_mm / dynamic_threshold_mm if dynamic_threshold_mm > 1e-9 else float("inf")

    lo, hi = 1.0 - hysteresis_band, 1.0 + hysteresis_band
    is_borderline = lo <= ratio <= hi

    if is_borderline:
        label = "FILL"
        reason = f"BORDERLINE (ratio={ratio:.2f} trong [{lo:.2f},{hi:.2f}]) -> mac dinh FILL"
    elif ratio <= 1.0:
        # --- BẢN FIX CHỐT HẠ V2: Phân biệt Nền đục lỗ và Text dính liền ---
        num_holes = len(holes)
        
        if num_holes >= 4 and solidity > 0.40:
            label = "FILL"
            reason = f"chua nhieu lo (holes={num_holes}) va dac (sol={solidity:.2f}>0.40) -> ep FILL (nen chua chu)"
        else:
            dynamic_ink_gate = ink_ratio_gate if is_hollow else 0.60
            
            if ink_ratio > dynamic_ink_gate and solidity > solidity_gate:
                label = "FILL"
                reason = f"duoi nguong nhung ink_ratio={ink_ratio:.2f}>{dynamic_ink_gate} (hollow={is_hollow}) & solidity={solidity:.2f}>{solidity_gate} -> ep FILL"
            else:
                label = "SATIN"
                reason = f"{thickness_mm:.3f} <= {dynamic_threshold_mm:.3f} mm"
    else:
        label = "FILL"
        reason = f"{thickness_mm:.3f} > {dynamic_threshold_mm:.3f} mm"


    is_mixed_width = False
    mixed_split_masks = None
    spread_ratio = p10 = p90 = None
    if skel is not None and n_skel_px > 20:
        skel_dist_mm_arr = dist[skel] * 2.0 * pixel_to_mm
        p10, p90 = np.percentile(skel_dist_mm_arr, [10, 90])
        spread_ratio = (p90 / p10) if p10 > 1e-6 else float("inf")
        if spread_ratio >= 1.8:
            is_mixed_width = True
            fill_m, satin_m = split_mixed_width_shape(mask_bool, dist, skel, pixel_to_mm, dynamic_threshold_mm)
            if fill_m is not None:
                mixed_split_masks = (fill_m, satin_m)

    return {
        "mask": shape_mask, "dist": dist, "max_dist_px": max_dist_px,
        "max_thickness_mm": max_thickness_mm, "skeleton_median_mm": skeleton_median_mm,
        "n_skel_px": n_skel_px, "minrect_short_mm": minrect_short_mm,
        "used_minrect": used_minrect, "thickness_source": thickness_source,
        "thickness_mm": thickness_mm, "solidity": solidity, "ink_ratio": ink_ratio,
        "ratio": ratio, "is_borderline": is_borderline, "label": label, "reason": reason,
        "is_mixed_width": is_mixed_width, "mixed_split_masks": mixed_split_masks,
        "p10": p10, "p90": p90, "spread_ratio": spread_ratio,
    }


def debug_shape_thickness_fixed(image_path, physical_width_mm=80.0, threshold_ratio=6.0,
                                 ink_ratio_gate=0.35, solidity_gate=0.45,
                                 hysteresis_band=0.15, solidity_trust_minrect=0.92,
                                 color_tolerance=25):
    """
    BAN v5 - TONG QUAT HOA CHO ANH PREVIEW NHIEU MAU / NHIEU VUNG
    """
    mode, fill_mask_full, satin_mask_full, total_fg = load_image_masks(image_path, color_tolerance)
    h, w = total_fg.shape
    pixel_to_mm = physical_width_mm / max(float(w), 1.0)

    ys, xs = np.where(total_fg)
    if len(xs) > 0:
        crop_w_px = xs.max() - xs.min() + 1
        logo_real_w_mm = crop_w_px * pixel_to_mm
    else:
        logo_real_w_mm = physical_width_mm

    dynamic_threshold_mm = logo_real_w_mm / threshold_ratio
    total_fg_area_px = float(np.count_nonzero(total_fg))

    print(f"\n=== THONG SO ANH & THRESHOLD THUC TE (MM) ===")
    print(f"Che do doc anh:              {mode}"
          + ("" if mode == "classified" else " (KHONG phai preview 2-mau chuan - dung fallback grayscale/Otsu)"))
    if mode == "classified":
        print(f"So pixel FILL (vang):        {int(fill_mask_full.sum())}")
        print(f"So pixel SATIN (hong):       {int(satin_mask_full.sum())}")
    print(f"Kich thuoc Canvas (Anh goc): {w}x{h} px ({physical_width_mm:.2f} mm)")
    print(f"Be ngang Logo that (Crop):   {logo_real_w_mm:.2f} mm")
    print(f"Ty le quy doi:               {pixel_to_mm:.6f} mm/px")
    print(f"Threshold (1/{threshold_ratio} logo):       {dynamic_threshold_mm:.3f} mm")
    print(f"Hysteresis band:             +-{hysteresis_band*100:.0f}%  "
          f"-> [{dynamic_threshold_mm*(1-hysteresis_band):.3f}, {dynamic_threshold_mm*(1+hysteresis_band):.3f}] mm")
    print(f"Solidity de tin minAreaRect: >= {solidity_trust_minrect}")
    print(f"===============================================\n")

    canvas_absolute = np.zeros((h, w), dtype=np.float32)
    canvas_relative = np.zeros((h, w), dtype=np.float32)
    shape_annotations = []
    n_mismatch = 0

    print(f"=== LOG KET QUA DO DAC THUC TE (MM) ===")
    valid_shape_count = 0

    color_groups = [("FILL", fill_mask_full), ("SATIN", satin_mask_full)] if mode == "classified" \
        else [("?", total_fg)]

    for current_color_label, color_mask in color_groups:
        if color_mask.sum() == 0:
            continue
        binary = (color_mask.astype(np.uint8)) * 255
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        if hierarchy is None:
            continue
        hierarchy = hierarchy[0]

        for i, contour in enumerate(contours):
            if hierarchy[i][3] != -1:
                continue
            if cv2.contourArea(contour) < 4:
                continue
            holes = [contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i]

            res = _measure_and_classify_shape(
                contour, holes, (h, w), pixel_to_mm, dynamic_threshold_mm,
                total_fg_area_px, ink_ratio_gate, solidity_gate,
                hysteresis_band, solidity_trust_minrect,
            )
            if res is None:
                continue

            valid_shape_count += 1
            mismatch = (mode == "classified" and res["label"] != current_color_label)
            if mismatch:
                n_mismatch += 1

            tag = f"[{current_color_label}]" if mode == "classified" else ""
            print(f"Shape {valid_shape_count} {tag} (color-group id: {i}):")
            print(f"  + Max do day:                  {res['max_thickness_mm']:.3f} mm")
            print(f"  + Med do day (qua SKELETON, {res['n_skel_px']} px): {res['skeleton_median_mm']:.3f} mm")
            print(f"  + minAreaRect short-side:      {res['minrect_short_mm']:.3f} mm")
            print(f"  + DUNG DE QUYET DINH ({res['thickness_source']}): {res['thickness_mm']:.3f} mm")
            print(f"  + solidity={res['solidity']:.2f}  ink_ratio={res['ink_ratio']:.2f}  ratio(dung/threshold)={res['ratio']:.2f}")
            print(f"  => KET LUAN (tinh lai tu dau): {res['label']}  ({res['reason']})")
            if mode == "classified":
                match_tag = "KHOP" if not mismatch else "*** LECH VOI MAU HIEN TAI TRONG ANH ***"
                print(f"  => Mau hien co trong anh: {current_color_label}  -> {match_tag}")
            if res["is_mixed_width"]:
                print(f"  [!] SHAPE HON HOP DO DAY: p10={res['p10']:.2f}mm p90={res['p90']:.2f}mm "
                      f"(chenh lech {res['spread_ratio']:.2f}x) -> 1 nhan duy nhat KHONG dai dien dung.")
                if res["mixed_split_masks"] is not None:
                    fill_m, satin_m = res["mixed_split_masks"]
                    split_preview_path = f"/home/claude/mixed_split_{current_color_label}_{i}.png"
                    sp = np.zeros((h, w, 3), dtype=np.uint8)
                    sp[fill_m] = (0, 255, 255)
                    sp[satin_m] = (255, 0, 255)
                    cv2.imwrite(split_preview_path, sp)
                    print(f"      -> Da tach va luu preview: {split_preview_path} "
                          f"(fill={int(fill_m.sum())}px, satin={int(satin_m.sum())}px)")
            print(f"  --------------------------------------------------")

            dist = res["dist"]
            canvas_absolute = np.maximum(canvas_absolute, dist)
            dist_norm_per_shape = dist / res["max_dist_px"]
            canvas_relative = np.maximum(canvas_relative, dist_norm_per_shape)

            _, _, _, max_loc = cv2.minMaxLoc(dist)
            ann_label = res["label"] + ("!" if mismatch else "")
            shape_annotations.append({'loc': max_loc, 'val': res["thickness_mm"], 'label': ann_label,
                                       'src': 'R' if res["used_minrect"] else 'S'})

    print(f"Tong so hinh khoi da quet: {valid_shape_count}")
    if mode == "classified":
        print(f"So shape co nhan TINH LAI khac voi mau dang co trong anh: {n_mismatch}")
    print(f"=======================================\n")

    masked_abs = np.ma.masked_where(canvas_absolute == 0, canvas_absolute)
    masked_rel = np.ma.masked_where(canvas_relative == 0, canvas_relative)

    display_bg = (total_fg.astype(np.uint8) * 255)

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    plt.title(f"1. Total foreground mask ({mode})")
    plt.imshow(display_bg, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Heatmap Tuyet doi\n(gia tri distance thuc)")
    im1 = plt.imshow(masked_abs, cmap='jet')
    plt.colorbar(im1, fraction=0.046, pad=0.04, label='Do day (Pixels)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Med-qua-Skeleton (khop pipeline that)\n"
               "'!' = nhan tinh lai LECH voi mau hien co trong anh")
    plt.imshow(display_bg, cmap='gray', alpha=0.2)
    im2 = plt.imshow(masked_rel, cmap='jet')
    plt.colorbar(im2, fraction=0.046, pad=0.04, label='Ty le (0.0 -> 1.0)')

    for ann in shape_annotations:
        x, y = ann['loc']
        is_mismatch_ann = ann['label'].endswith("!")
        color = 'red' if is_mismatch_ann else ('yellow' if ann['label'].startswith('FILL') else 'cyan')
        plt.plot(x, y, 'x', color=color, markersize=6)
        plt.text(x + 5, y, f"{ann['val']:.2f}mm[{ann['src']}] [{ann['label']}]", color='white',
                 fontsize=9, weight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=1))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Test thu
if __name__ == "__main__":
    debug_shape_thickness_fixed('data/opencv_test/predictions/test_pred.png', physical_width_mm=80.0, threshold_ratio=6.0)
