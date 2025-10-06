# fault Detection Simulation Orchestrator

این ریپو شامل یک معماری microservice برای اجرای grid از pipelineها (120 ترکیب) است:
- فاز 1: استخراج فیچر (preprocess → detectors) و ذخیره در `/data/features` (NPZ)
- فاز 2: orchestration از طریق HTTP (matcher, homography, shift, vae)
- فاز 3: جمع‌آوری نتایج و تولید گزارش (aggregator)

## پیش‌نیازها
- Docker + Docker Compose
- اگر VAE را با GPU اجرا می‌کنی: درایور NVIDIA و runtime docker مناسب نصب باشند.

## مراحل سریع

1. کلون ریپو و آماده‌سازی فولدرها:
```bash
git clone <repo>
cd crack-detection-simulation
mkdir -p data/raw data/features results logs
# پر کن data/raw با چند تصویر نمونه و ایجاد data/manifest.json
