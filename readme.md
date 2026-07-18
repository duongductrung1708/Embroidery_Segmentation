# Yêu cầu file SVG đầu vào — Stitch Classifier (Satin vs Fill)

Module nhận vào một file SVG và tự động phân loại từng vùng thành **satin**
hoặc **fill**. Vì thuật toán đọc và phân loại **trực tiếp trên hình học của
từng `<path>`** (không qua bước render/gộp màu thủ công), chất lượng và độ
"sạch" của file SVG đầu vào quyết định trực tiếp độ chính xác của kết quả.
Tài liệu này liệt kê các yêu cầu bắt buộc và khuyến nghị cho file SVG.

---

## 1. Kích thước vật lý phải khai báo đúng

Ngưỡng phân biệt satin/fill được tính **động** dựa trên bề rộng thật (mm)
của logo:

```python
threshold_mm = bề_rộng_thật_của_logo_mm / 6
```

Nét nào **mỏng hơn** ngưỡng này → khả năng cao là **satin**; nét nào
**dày hơn** → **fill**. Vì vậy nếu kích thước vật lý bị khai báo sai, toàn
bộ ngưỡng phân loại sẽ lệch theo.

Cách hệ thống suy ra kích thước, theo thứ tự ưu tiên:

1. Thuộc tính `width` trên thẻ `<svg>`, kèm đơn vị hợp lệ: `mm`, `cm`,
   `in`, hoặc không đơn vị/`px`/`pt` (quy đổi theo 96 dpi).
2. Nếu không có `width`, suy ra từ `viewBox` (giả định 96 dpi).
3. Nếu cả hai đều thiếu → mặc định 80mm — **gần như chắc chắn sai** với
   logo thật.

**Yêu cầu:** mọi file SVG upload phải có `width` (kèm đơn vị đo lường thật)
hoặc `viewBox` đúng tỉ lệ thật của thiết kế.

---

## 2. File SVG phải "sạch" về mặt hình học

- Mỗi vùng cần phân loại phải là một `<path>` render được bình thường
  (không lỗi cú pháp path, không phụ thuộc tài nguyên ngoài).
- **Không đặt path thực tế của thiết kế bên trong `<defs>`, `<clipPath>`,
  hoặc `<mask>`** — các path trong những thẻ này sẽ bị hệ thống bỏ qua
  hoàn toàn, coi như không tồn tại.
- **Lỗ hổng bên trong hình** (ví dụ thân chữ "e", "a", "o", hoặc logo có
  viền rỗng ở giữa) **phải được tạo bằng chính fill-rule của path đó**
  (path tự đục lỗ bằng sub-path ngược chiều, `evenodd`/`nonzero`), **không
  được** giả lập lỗ bằng cách vẽ đè một path màu nền/trắng lên trên. Hệ
  thống tách hình dựa trên contour hierarchy thật của path — nếu lỗ là giả
  (chỉ là 2 path chồng lên nhau), kết quả phân loại sẽ sai.

- **Không để sót path rác/artefact** (các mảnh vụn cực nhỏ do phần mềm
  vector hoá tự sinh ra). Những path quá nhỏ (dưới ngưỡng diện tích
  ~0.2mm² và bề dày dưới ~0.4mm) sẽ bị coi là nhiễu và tự động loại bỏ,
  nhưng càng ít rác thì kết quả càng ổn định — không nên phụ thuộc vào cơ
  chế lọc nhiễu này.
- Mỗi `<path>` nên đại diện cho **một vùng có ý nghĩa hình học riêng** (một
  mảng satin hoặc fill). Tránh gộp nhiều vùng không liên quan vào chung
  một path bằng nhiều sub-path rời rạc — hệ thống phân loại theo cụm hình
  học trong cùng một path.
- Không nên áp dụng phép biến đổi (transform) phức tạp/lồng nhau nhiều lớp
  gây méo tỉ lệ thật của path so với kích thước đã khai báo ở mục 1 — bề
  dày đo được phải phản ánh đúng bề dày thật ngoài đời của nét.
- **Tránh dùng màu tô dạng gradient** (`linearGradient`, `radialGradient`,
  `fill="url(#...)"`) cho path — nên dùng **màu đặc, không trong suốt**
  (`fill="#RRGGBB"`, `fill-opacity="1"`). Gradient có 2 vấn đề: (1) không
  còn là một mã màu duy nhất, nên phá vỡ quy ước gán nhãn theo màu ở mục 5
  (hệ thống không nhận ra `#ff0000`/`#0000ff` khi màu là một chuỗi tham
  chiếu gradient); (2) nếu gradient có các điểm dừng trong suốt/mờ dần,
  vùng mờ đó có thể bị tính sai là "không có mực" ở bước xác định biên
  logo, làm lệch kích thước thật dùng để tính ngưỡng phân loại cho toàn bộ
  file. Ngoài ra, chỉ thêu được màu chỉ đặc — gradient vốn không có ý
  nghĩa vật lý khi lên khung thêu.

---

## 3. Tránh path có bề dày nằm sát ngưỡng phân loại

Vì hệ thống chỉ dựa vào **một con số ngưỡng duy nhất**
(`threshold_mm = bề_rộng_logo / 6`) để quyết định satin hay fill, những
path có bề dày nằm **quá sát ngưỡng này** là nguồn gây sai nhiều nhất:

- Một nét satin nhưng vẽ hơi dày (gần chạm ngưỡng) có thể bị nhận nhầm
  thành fill.
- Một mảng fill nhưng vẽ hơi mỏng (gần chạm ngưỡng) có thể bị nhận nhầm
  thành satin.

Vì vậy khi thiết kế/
xuất SVG, cần:

- **Chủ đích hoá bề dày của từng path**, tránh vẽ các nét/mảng có bề dày
  rơi đúng vào khoảng "lưng chừng" giữa nét satin điển hình và mảng fill
  điển hình của cùng một logo. Nếu một chi tiết về bản chất là satin, nên
  vẽ mỏng rõ ràng so với ngưỡng; nếu về bản chất là fill, nên vẽ dày rõ
  ràng so với ngưỡng — càng cách xa ngưỡng, kết quả càng ổn định.
- Trong cùng một logo, **giữ sự nhất quán về bề dày giữa các path cùng
  loại**: mọi nét satin nên có bề dày tương đồng nhau, mọi mảng fill nên
  rõ ràng dày hơn hẳn — tránh tình trạng 2 path cùng ý đồ là satin nhưng
  một cái mỏng, một cái dày gần gấp đôi, khiến một trong hai bị lệch nhãn.
- Nếu một chi tiết trong thiết kế **thực sự nằm ở ranh giới** (ví dụ do
  yêu cầu thẩm mỹ không thể vẽ mỏng/dày hơn), nên **gắn nhãn tường minh**
  cho path đó (xem mục 4 — `inkscape:label="satin"`/`"fill"`) thay vì để
  hệ thống tự đoán bằng hình học, vì hình học ở vùng biên là không đáng
  tin cậy.
- Với các mảng fill bị đục nhiều lỗ nhỏ (dạng lưới, dạng hoa văn) khiến bề
  dày đo được giữa các lỗ trở nên rất mỏng: nên đảm bảo mảng đó vẫn đủ đặc
  (diện tích liền khối lớn so với tổng lượng mực) để không bị hiểu nhầm
  thành một viền satin mảnh — hệ thống có xử lý trường hợp này (mảng nhiều
  lỗ + đặc → vẫn coi là fill) nhưng độ tin cậy giảm khi hình dạng thực sự
  mơ hồ.

Nói ngắn gọn: **thiết kế càng "rõ ràng" giữa hai thái cực mảnh–satin và
dày–fill thì kết quả càng chính xác; phần khó nhất luôn nằm ở những chi
tiết cỡ trung, gần bằng đúng 1/6 bề rộng logo.**

---

## 4. Cẩn trọng khi các path satin tiếp xúc hoặc chồng lên nhau

Hai path (dù cùng là satin, hay khác loại) **chạm biên hoặc chồng mép lên
nhau** cũng là một nguồn gây sai khác, tách biệt với vấn đề bề dày ở mục 3:

- **Nếu hai nét vốn là một khối satin liên tục nhưng bị tách thành hai
  `<path>` riêng, đặt sát/chồng mép nhau** để "trông như dính liền": hệ
  thống xử lý từng `<path>` độc lập, nên mỗi nét vẫn được đo bề dày *riêng
  lẻ theo đúng path của nó*, chứ không tự động gộp lại thành một khối để
  đo chung. Nếu ý đồ thiết kế là một satin liền mạch, hãy vẽ **gộp thành
  một `<path>` duy nhất** — tách rời thành nhiều path sát nhau chỉ nên
  dùng khi các phần đó thực sự là các đối tượng độc lập.
- **Nếu một path satin nằm áp sát/chạm biên vào bên trong một path satin
  khác có lỗ rỗng** (dạng viền/outline), hệ thống có một bước tinh chỉnh
  theo ngữ cảnh: nếu path bên trong bị bao phủ ≥ 50% diện tích bởi phần
  rỗng của path viền đó, **và** có chạm biên viền, path bên trong sẽ **tự
  động bị ép thành fill** (cơ chế này vốn để xử lý đúng phần thân bên
  trong các chữ cái như "e", "a", "o"). Nếu hai nét satin của bạn tiếp xúc
  nhau nhưng **không** có quan hệ dạng "một cái nằm trong lỗ của cái kia",
  quy tắc này sẽ không áp dụng — nhưng nếu vô tình rơi đúng vào tình huống
  đó (một satin nhỏ lọt gần hết vào phần rỗng của một satin viền lớn và
  có điểm chạm), nó **sẽ** bị đổi thành fill dù ý đồ ban đầu là satin.
  Nếu không muốn bị ép nhãn theo ngữ cảnh này, tránh để một path satin nằm
  lọt gần hết bên trong lỗ của một path satin khác, hoặc tách khoảng cách
  rõ ràng thay vì để chạm biên.
- **Hai path chồng mép lên nhau (overlap thật sự, không chỉ chạm biên)**
  dù không liên quan đến quy tắc trên vẫn gây rủi ro: ở bước vẽ nhãn màu
  cuối cùng, các path được vẽ chồng lên nhau theo đúng thứ tự xuất hiện
  trong file SVG — path nào nằm sau trong file sẽ **đè màu/nhãn lên** phần
  pixel chồng lấn của path nằm trước. Nếu hai path vô tình overlap ở biên
  (thường do xuất file từ phần mềm vector không khít tuyệt đối), phần
  biên đó sẽ hiển thị nhãn của path vẽ sau, có thể không đúng ý đồ thiết
  kế ban đầu.

**Khuyến nghị:** với các nét/mảng thực sự là những đối tượng tách biệt,
nên để một khoảng hở nhỏ rõ ràng giữa chúng thay vì để chạm khít biên hay
chồng mép; với các nét vốn là một khối liền mạch, nên gộp thành một
`<path>` duy nhất ngay từ khâu thiết kế thay vì tách rời rồi ghép lại bằng
mắt thường.

---

## 5. (Tùy chọn) Gán nhãn có sẵn cho path

Nếu file SVG đã có sẵn nhãn (ví dụ do người thiết kế gắn tay), hệ thống có
thể ưu tiên đọc trực tiếp thay vì tự suy luận hình học, theo thứ tự:

1. Thuộc tính `inkscape:label` (hoặc thuộc tính nào có tên kết thúc bằng
   `label`) chứa chuỗi `"satin"` hoặc `"fill"` (không phân biệt hoa
   thường).
2. Thuộc tính `data-label`, `data-stitch`, `data-stitch-type`, `class`,
   hoặc `id` chứa chuỗi `"satin"` hoặc `"fill"`.
3. Mã màu trong `style`: `#ff0000` → satin; `#0000ff`/`#00ff00` → fill.
   (Chỉ nhận diện được màu đặc dạng mã hex — path dùng gradient sẽ không
   khớp được với quy ước này, xem lưu ý ở mục 2.)
4. Nếu path không có nhãn nào ở trên, hệ thống sẽ tự phân loại hoàn toàn
   dựa trên hình học (bề dày, độ đặc, tỉ lệ mực) như mô tả ở mục 6.

Đây không phải yêu cầu bắt buộc — file SVG không có nhãn vẫn được xử lý
bình thường bằng thuật toán hình học.

---

## 6. Vì sao thuật toán ra quyết định như vậy

Với mỗi path, hệ thống đo:

- **Bề dày (thickness_mm):** bề dày ước lượng của nét/mảng.
- **Độ đặc (solidity):** diện tích thật / diện tích convex hull.
- **Tỉ lệ mực (ink_ratio):** diện tích shape / tổng diện tích có mực toàn
  path — đo mức độ shape chiếm phần lớn lượng mực.
- **Số lỗ, tỉ lệ khung hình, có phải viền ngoài lớn nhất hay không.**

Quy tắc:

1. Bề dày quá nhỏ **và** diện tích quá nhỏ → coi là **nhiễu**, loại bỏ.
2. Nét **mỏng hơn ngưỡng** (ứng viên satin), nhưng sẽ bị ép thành **fill**
   nếu:
   - có nhiều lỗ (≥ 4) **và** khá đặc (không phải chữ/text mảnh), hoặc
   - chiếm tỉ lệ mực lớn **và** khá đặc (mảng lớn giả dạng viền mỏng).
   - Còn lại → **satin**.
3. Nét **dày hơn ngưỡng** → **fill**.
4. **Tinh chỉnh theo ngữ cảnh:** phần satin nằm gọn bên trong một satin
   khác có lỗ và chạm biên (ví dụ thân bên trong chữ "e", "a", "o") sẽ
   được ép lại thành **fill**.

Vì toàn bộ quyết định dựa thuần trên hình học (không hiểu ngữ nghĩa thiết
kế), **SVG càng sạch, càng đúng tỉ lệ thật thì kết quả càng chính xác.**

---

## 7. Checklist nhanh trước khi upload SVG

- [ ] Có `width` (kèm đơn vị thật) hoặc `viewBox` đúng tỉ lệ thật của
      thiết kế.
- [ ] Không có path thực tế nào nằm trong `<defs>`, `<clipPath>`, `<mask>`.
- [ ] Lỗ trong hình được tạo bằng fill-rule của path, không vẽ đè path
      màu nền lên trên.
- [ ] Không còn path rác/artefact siêu nhỏ sót lại từ phần mềm vector hoá.
- [ ] Không có transform gây méo tỉ lệ thật của thiết kế.
- [ ] Fill của mọi path là màu đặc, không dùng gradient hoặc độ trong suốt
      mờ dần.
- [ ] Bề dày mỗi path đủ rõ ràng — nét satin vẽ mỏng hẳn, mảng fill vẽ dày
      hẳn so với ngưỡng `bề_rộng_logo / 6`, tránh rơi vào vùng lưng chừng.
- [ ] Các nét vốn là một khối satin liền mạch được gộp thành một `<path>`
      duy nhất, không tách rời rồi đặt sát/chồng mép nhau.
- [ ] Không có path chồng mép (overlap) ngoài ý muốn ở biên giữa các
      vùng khác nhãn; các đối tượng tách biệt có khoảng hở rõ ràng thay
      vì chạm khít biên.
- [ ] (Nếu có) nhãn `inkscape:label`/`class`/`id` đặt đúng, rõ ràng, tránh
      nhầm với thuộc tính khác cùng tên; ưu tiên gắn nhãn tường minh cho
      các chi tiết nằm sát ranh giới satin/fill.

---

## 8. Giới hạn cần lưu ý

- Ngưỡng `threshold_mm = bề_rộng_logo / 6` là tỉ lệ cố định — logo có chi
  tiết đặc biệt mảnh hoặc đặc biệt dày so với tổng thể có thể cần tinh
  chỉnh riêng, không đảm bảo đúng 100% cho mọi phong cách thiết kế.
- Bước phân loại hình học chạy trên canvas thu nhỏ để tăng tốc — chi tiết
  cực nhỏ có thể mất độ chính xác nhẹ; chỉ bước render nhãn cuối cùng mới
  ở độ phân giải đầy đủ.
- Hệ thống không hiểu ngữ nghĩa thiết kế (chữ, logo, icon...) — mọi quyết
  định đều dựa trên số đo hình học thuần túy.
- Cơ chế tinh chỉnh theo ngữ cảnh cho phần "nằm trong lỗ của viền satin"
  (mục 4) chỉ xử lý đúng khi quan hệ chứa/chạm biên giữa hai path là rõ
  ràng; các trường hợp tiếp xúc mập mờ (chạm một phần nhỏ, bao phủ gần
  nhưng không đủ 50%...) có thể cho kết quả không như mong đợi và nên
  kiểm tra lại bằng mắt trên ảnh preview.
