import cv2
import numpy as np
import matplotlib.pyplot as plt

try:
    from skimage.morphology import skeletonize
except ImportError:
    skeletonize = None
    print("[CANH BAO] Chua cai 'scikit-image' -> pip install scikit-image. "
          "Neu khong co, script se fallback ve median-toan-mask (SAI, khong khop pipeline that).")


def debug_shape_thickness_fixed(image_path, physical_width_mm=80.0, threshold_ratio=6.0,
                                 ink_ratio_gate=0.35, solidity_gate=0.45,
                                 hysteresis_band=0.15, solidity_trust_minrect=0.92):
    """
    BAN SUA LOI (v3):
    ------------------
    1) [BUG #1] Median tinh tren TOAN BO pixel mask -> bi keo tut vi da so
       dien tich nam gan bien. SUA: median CHI TREN SKELETON.

    2) [BO SUNG] Them lop ink_ratio+solidity (chan nen-gia-vien) va
       hysteresis-band, khop 100% voi production.

    3) [BUG #2 - MOI] Ngay ca median-qua-skeleton cung KHONG BAT BIEN theo
       chieu dai shape: voi hinh co 2 dau vat/nhon (vd hinh binh hanh vat
       cheo), doan vat co DO DAI TUYET DOI gan nhu co dinh, nhung chiem TY
       TRONG % khac nhau tren tong chieu dai skeleton tuy shape ngan/dai.
       -> 2 shape rong BANG NHAU nhung dai khac nhau se cho ra median khac
       han nhau (shape ngan bi keo tut manh hon). Da kiem chung thuc te:
       3 cot SURGE rong ~14.6-14.8mm (giong het nhau qua minAreaRect &
       max_dist) nhung median-qua-skeleton lai ra 6.86 / 9.96 / 14.32mm.

       SUA: voi shape GAN LOI TUYET DOI (solidity cao, khong lo) - dung
       `cv2.minAreaRect` short-side lam thuoc do be rong. Day la phep do
       hinh hoc thuan tuy, BAT BIEN hoan toan theo chieu dai shape, khong
       bi anh huong boi vung vat dau. Chi giu skeleton-median cho shape
       loi thap (chu cai, net cong, co lom) vi luc do minAreaRect khong
       con dai dien cho be rong net.
    """
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[LOI] Khong tim thay anh: {image_path}")
        return

    h, w = img.shape[:2]
    pixel_to_mm = physical_width_mm / max(float(w), 1.0)

    if len(img.shape) == 3 and img.shape[2] == 4:
        _, binary = cv2.threshold(img[:, :, 3], 127, 255, cv2.THRESH_BINARY)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        corners_sum = int(binary[0, 0]) + int(binary[0, -1]) + int(binary[-1, 0]) + int(binary[-1, -1])
        if corners_sum > 255 * 2:
            binary = cv2.bitwise_not(binary)

    ys, xs = np.where(binary > 0)
    if len(xs) > 0:
        crop_w_px = xs.max() - xs.min() + 1
        logo_real_w_mm = crop_w_px * pixel_to_mm
    else:
        logo_real_w_mm = physical_width_mm

    dynamic_threshold_mm = logo_real_w_mm / threshold_ratio
    total_fg_area_px = float(np.count_nonzero(binary))

    print(f"\n=== THONG SO ANH & THRESHOLD THUC TE (MM) ===")
    print(f"Kich thuoc Canvas (Anh goc): {w}x{h} px ({physical_width_mm:.2f} mm)")
    print(f"Be ngang Logo that (Crop):   {logo_real_w_mm:.2f} mm")
    print(f"Ty le quy doi:               {pixel_to_mm:.6f} mm/px")
    print(f"Threshold (1/{threshold_ratio} logo):       {dynamic_threshold_mm:.3f} mm")
    print(f"Hysteresis band:             +-{hysteresis_band*100:.0f}%  "
          f"-> [{dynamic_threshold_mm*(1-hysteresis_band):.3f}, {dynamic_threshold_mm*(1+hysteresis_band):.3f}] mm")
    print(f"Solidity de tin minAreaRect: >= {solidity_trust_minrect}")
    print(f"===============================================\n")

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        print("Khong tim thay hinh khoi nao trong anh.")
        return
    hierarchy = hierarchy[0]

    canvas_absolute = np.zeros((h, w), dtype=np.float32)
    canvas_relative = np.zeros((h, w), dtype=np.float32)
    shape_annotations = []

    print(f"=== LOG KET QUA DO DAC THUC TE (MM) ===")
    valid_shape_count = 0

    for i, contour in enumerate(contours):
        if hierarchy[i][3] != -1:
            continue
        if cv2.contourArea(contour) < 4:
            continue

        holes = [contours[j] for j, hj in enumerate(hierarchy) if hj[3] == i]
        is_hollow = len(holes) > 0

        shape_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(shape_mask, [contour], -1, 255, thickness=cv2.FILLED)
        for hole in holes:
            cv2.drawContours(shape_mask, [hole], -1, 0, thickness=cv2.FILLED)

        dist = cv2.distanceTransform(shape_mask, cv2.DIST_L2, 5)
        max_dist_px = np.max(dist)
        if max_dist_px == 0:
            continue

        max_thickness_mm = (max_dist_px * 2.0) * pixel_to_mm

        median_thickness_mm = max_thickness_mm
        n_skel_px = 0
        if skeletonize is not None:
            skel = skeletonize(shape_mask > 0)
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

        skeleton_median_mm = median_thickness_mm  # luu truoc khi bi ghi de boi minAreaRect (neu co)

        # --- [MOI] minAreaRect short-side cho shape gan loi ---
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

        valid_shape_count += 1
        thickness_mm = median_thickness_mm
        ratio = thickness_mm / dynamic_threshold_mm if dynamic_threshold_mm > 1e-9 else float("inf")

        lo, hi = 1.0 - hysteresis_band, 1.0 + hysteresis_band
        is_borderline = lo <= ratio <= hi

        if is_borderline:
            label = "FILL"
            reason = f"BORDERLINE (ratio={ratio:.2f} trong [{lo:.2f},{hi:.2f}]) -> mac dinh FILL"
        elif ratio <= 1.0:
            if ink_ratio > ink_ratio_gate and solidity > solidity_gate:
                label = "FILL"
                reason = f"duoi nguong nhung ink_ratio={ink_ratio:.2f}>{ink_ratio_gate} & solidity={solidity:.2f}>{solidity_gate} -> ep FILL (nen gia vien)"
            else:
                label = "SATIN"
                reason = f"{thickness_mm:.3f} <= {dynamic_threshold_mm:.3f} mm"
        else:
            label = "FILL"
            reason = f"{thickness_mm:.3f} > {dynamic_threshold_mm:.3f} mm"

        print(f"Shape {valid_shape_count} (ID Contour: {i}):")
        print(f"  + Max do day:                  {max_thickness_mm:.3f} mm")
        print(f"  + Med do day (qua SKELETON, {n_skel_px} px): {skeleton_median_mm:.3f} mm")
        print(f"  + minAreaRect short-side:      {minrect_short_mm:.3f} mm")
        print(f"  + DUNG DE QUYET DINH ({thickness_source}): {thickness_mm:.3f} mm")
        print(f"  + solidity={solidity:.2f}  ink_ratio={ink_ratio:.2f}  ratio(dung/threshold)={ratio:.2f}")
        print(f"  => KET LUAN: {label}  ({reason})")
        print(f"  --------------------------------------------------")

        canvas_absolute = np.maximum(canvas_absolute, dist)
        dist_norm_per_shape = dist / max_dist_px
        canvas_relative = np.maximum(canvas_relative, dist_norm_per_shape)

        _, _, _, max_loc = cv2.minMaxLoc(dist)
        shape_annotations.append({'loc': max_loc, 'val': thickness_mm, 'label': label,
                                   'src': 'R' if used_minrect else 'S'})

    print(f"Tong so hinh khoi da quet: {valid_shape_count}")
    print(f"=======================================\n")

    masked_abs = np.ma.masked_where(canvas_absolute == 0, canvas_absolute)
    masked_rel = np.ma.masked_where(canvas_relative == 0, canvas_relative)

    plt.figure(figsize=(18, 8))

    plt.subplot(1, 3, 1)
    plt.title("1. Mask (nen den, hinh trang)")
    plt.imshow(binary, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title("2. Heatmap Tuyet doi\n(gia tri distance thuc)")
    im1 = plt.imshow(masked_abs, cmap='jet')
    plt.colorbar(im1, fraction=0.046, pad=0.04, label='Do day (Pixels)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title("3. Med-qua-Skeleton (khop pipeline that)\n"
               "so o day la thickness dung de RA QUYET DINH")
    plt.imshow(binary, cmap='gray', alpha=0.2)
    im2 = plt.imshow(masked_rel, cmap='jet')
    plt.colorbar(im2, fraction=0.046, pad=0.04, label='Ty le (0.0 -> 1.0)')

    for ann in shape_annotations:
        x, y = ann['loc']
        color = 'yellow' if ann['label'] == 'FILL' else 'cyan'
        plt.plot(x, y, 'x', color=color, markersize=6)
        plt.text(x + 5, y, f"{ann['val']:.2f}mm[{ann['src']}] [{ann['label']}]", color='white',
                 fontsize=9, weight='bold', bbox=dict(facecolor='black', alpha=0.6, pad=1))

    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Test thu
debug_shape_thickness_fixed('data/opencv_test/predictions/test_pred.png', physical_width_mm=80.0, threshold_ratio=6.0)