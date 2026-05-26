# Segmentor 示例性能测试报告

## 1. 测试说明

本文档记录 `example/seg` 分割示例在 UNet 模型上的推理性能表现。

测试输入尺寸统一为：

| 参数 | 值 |
| --- | --- |
| 输入高度 | `512` |
| 输入宽度 | `512` |
| Batch Size | `1` |

测试资源如下：

| 项目 | 路径 |
| --- | --- |
| FP32 TensorRT Engine | `unet.plan` |
| FP16 TensorRT Engine | `unet_fp16.plan` |
| 源 ONNX 模型 | `unet.onnx` |
| 测试图片 | `imgs/car.jpg` |
| 配置文件 | `config/seg_config.json` |
| 输出结果图 | `imgs/car_res.jpg` |
| FP16 输出结果图 | `imgs/car_res_fp16.jpg` |

性能指标说明：

| 指标 | 含义 |
| --- | --- |
| 平均推理耗时 | 单张图片端到端分割推理平均耗时，单位为 `ms` |
| FPS | 按平均推理耗时换算得到的吞吐率，计算方式为 `1000 / 平均推理耗时(ms)` |
| 前景像素数 | 分割结果中非背景像素数量 |

> 注：性能数据来自 `example/seg` 示例程序输出结果，主要用于记录当前 UNet 分割示例的端到端推理性能。

## 2. FP32 性能结果

| 模型 | 输入尺寸 | 原图尺寸 | 平均推理耗时 | 最小推理耗时 | 最大推理耗时 | FPS | 前景像素数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| UNet | `512 x 512` | `1280 x 853` | `13.614437 ms` | `13.207452 ms` | `14.550836 ms` | `73.451440` | `432005` |

## 3. FP16 性能结果

| 模型 | 输入尺寸 | 原图尺寸 | 平均推理耗时 | 最小推理耗时 | 最大推理耗时 | FPS | 前景像素数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| UNet | `512 x 512` | `1280 x 853` | `7.281235 ms` | `7.127803 ms` | `7.655472 ms` | `137.339334` | `430924` |

## 4. 性能对比

| 模型 | FP32 平均耗时 | FP16 平均耗时 | 耗时下降 | FP32 FPS | FP16 FPS | FPS 提升 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet | `13.614437 ms` | `7.281235 ms` | `46.5%` | `73.451440` | `137.339334` | `87.0%` |

## 5. 输出结果

示例程序完成推理后会生成：

| 输出文件 | 说明 |
| --- | --- |
| `imgs/car_res.jpg` | 将分割 mask 以半透明绿色覆盖在原图上的三通道 JPEG 图片 |
| `imgs/car_res_fp16.jpg` | FP16 推理生成的半透明分割结果图 |

## 6. 性能结论

- 在 `512 x 512` 模型输入尺寸下，UNet 分割示例平均推理耗时为 `13.614437 ms`。
- FP16 平均推理耗时为 `7.281235 ms`，相比 FP32 下降约 `46.5%`。
- FP16 吞吐率为 `137.339334 FPS`，相比 FP32 提升约 `87.0%`。
- 输出 mask 尺寸与原图一致，为 `1280 x 853`。
- `tools/predict_onnx.py` 与 `example/seg` 示例项目的推理结果已对齐，半透明叠加结果保持一致。

## 7. 运行示例

可使用如下命令运行 FP32 分割示例并输出性能指标：

```bash
./example/seg/build/seg_sdk_sample unet.plan imgs/car.jpg config/seg_config.json imgs/car_res.jpg 5 50
```

可使用如下命令运行 FP16 分割示例并输出性能指标：

```bash
./example/seg/build/seg_sdk_sample unet_fp16.plan imgs/car.jpg config/seg_config_fp16.json imgs/car_res_fp16.jpg 5 50
```

参数说明：

| 参数位置 | 示例值 | 说明 |
| --- | --- | --- |
| 1 | `unet.plan` / `unet_fp16.plan` | 分割模型路径 |
| 2 | `imgs/car.jpg` | 输入图片路径 |
| 3 | `config/seg_config.json` / `config/seg_config_fp16.json` | 分割配置文件路径 |
| 4 | `imgs/car_res.jpg` / `imgs/car_res_fp16.jpg` | 分割结果图片保存路径 |
| 5 | `5` | 预热次数 |
| 6 | `50` | 性能统计次数 |
