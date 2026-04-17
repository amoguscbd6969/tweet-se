# Yugioh Rarity Multitask Baseline

Baseline này triển khai đúng hướng bạn yêu cầu:

- Feature engineering từ `rarity` sang **8 binary cues** để hỗ trợ Task 1.
- Giữ nguyên 3 nhãn official: `name_foil`, `art_foil`, `full_foil` cho Task 2/3/4.
- Tạo **3 dataset views riêng**:
  - `name` crop (vùng tên lá bài)
  - `art` crop (vùng ảnh trung tâm)
  - `full` (toàn bộ card)
- Dùng **3 augmentation policy riêng** tương ứng với 3 view để học đúng tín hiệu:
  - Name: đẩy mạnh color shift để phân biệt bạc/vàng/7 màu.
  - Art: nhấn vào texture/contrast để bắt sparkle + sọc.
  - Full: nhấn tone/CLAHE để bắt hiệu ứng full foil và 25th mark.
- Huấn luyện 11 binary models (ResNet18 + EfficientNetB4 ensemble) theo yêu cầu.
- Decode lại rarity bằng hard-rule + consistency check giữa Task 1 và Task 2/3/4.

## Cài đặt

```bash
pip install torch torchvision timm albumentations opencv-python pandas scikit-learn
```

## Input train CSV

`train.csv` cần có:
- `id`: tên file ảnh, ví dụ `card10001.jpg`
- `rarity`: label 1..8

Script sẽ tự sinh các nhãn phụ từ `rarity`.

## Train baseline

```bash
python baseline_multitask.py \
  --train_csv /path/to/train.csv \
  --image_dir /path/to/train_images \
  --out_dir outputs
```

Output:
- `outputs/*_ensemble.pt`: trọng số cho từng task
- `outputs/cv_metrics.json`: metric CV + weighted score

## Notes

- TTA có trong hàm `infer_ensemble`.
- SWA đã tích hợp trong vòng train.
- Có thể tăng `epochs`, `image_size`, `batch_size` trong `TrainConfig` để cải thiện score.
