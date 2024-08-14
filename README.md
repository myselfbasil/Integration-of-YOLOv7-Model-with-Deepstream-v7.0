# Integration of YOLOv7 Model with Deepstream v7.0

Reference Page: https://github.com/myselfbasil/DeepStream-Yolo

Here, I will be demonstrating the entire process for running a deepstream app for Fall Detection

Note: For other versions of yolo models also, follow the same proceedures :)

# **1. Initial Setup:**

**Download the YOLOv7 repo and install the requirements**

```bash
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
pip3 install onnx onnxsim onnxruntime
```

**Copy appropriate conversor file for your model from the link her**

https://github.com/myselfbasil/DeepStream-Yolo/tree/master/utils

Here, Copy the `export_yoloV7.py` file from `DeepStream-Yolo/utils` directory to the `yolov7` folder.

# **2. Download the model**

Note: If you have your own model, paste it or download the `.pt` file from [YOLOv7](https://github.com/WongKinYiu/yolov7/releases/) releases (example for YOLOv7) 

```bash
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

**Reparameterize your model (for custom models)**

Custom YOLOv7 models cannot be directly converted to engine file. Therefore, you will have to reparameterize your model using the code [here](https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb). Make sure to convert your custom checkpoints in YOLOv7 repository, and then save your reparmeterized checkpoints for conversion in the next step.

# **3. Convert the Model:**

Generate the ONNX model file (example for YOLOv7)

```bash
python3 export_yoloV7.py -w yolov7.pt --dynamic
```

**NOTE**: To convert a P6 model

```bash
--p6
```

**NOTE**: To change the inference size (defaut: 640 / 1280 for `--p6` models)

```bash
-s SIZE
--size SIZE
-s HEIGHT WIDTH
--size HEIGHT WIDTH
```

Example for 1280

```bash
-s 1280
```

or

```bash
-s 1280 1280
```

**NOTE**: To simplify the ONNX model (DeepStream >= 6.0)

```bash
--simplify
```

**NOTE**: To use dynamic batch-size (DeepStream >= 6.1)

```bash
--dynamic
```

**NOTE**: To use static batch-size (example for batch-size = 4)

```bash
--batch 4
```

**NOTE**: If you are using the DeepStream 5.1, remove the `--dynamic` arg and use opset 12 or lower. The default opset is 12.

```bash
--opset 12
```

Copy the generated ONNX model file and labels.txt file (if generated) to the `DeepStream-Yolo` folder.

# **4. Set CUDA Version & Compile the lib**

1. Open the `DeepStream-Yolo` folder and compile the lib
2. Set the `CUDA_VER` according to your DeepStream version

```bash
nvcc --version
```

```bash
export CUDA_VER=XY.Z
```

- x86 platform
    
    ```bash
    DeepStream 7.0 / 6.4 = 12.2
    DeepStream 6.3 = 12.1
    DeepStream 6.2 = 11.8
    DeepStream 6.1.1 = 11.7
    DeepStream 6.1 = 11.6
    DeepStream 6.0.1 / 6.0 = 11.4
    DeepStream 5.1 = 11.1
    ```
    
- Jetson platform
    
    ```bash
    DeepStream 7.0 / 6.4 = 12.2
    DeepStream 6.3 / 6.2 / 6.1.1 / 6.1 = 11.4
    DeepStream 6.0.1 / 6.0 / 5.1 = 10.2
    ```
    
- Compile the lib

```bash
make -C nvdsinfer_custom_impl_Yolo clean && make -C nvdsinfer_custom_impl_Yolo
```

## 6. Making changes to the Configuration Files

**Edit the config_infer_primary_yoloV7 file**

Edit the `config_infer_primary_yoloV7.txt` file according to your model (example for YOLOv7 with 80 classes)

```bash
[property]
...
onnx-file=/path/to/your/.onnx_file
...
num-detected-classes=80
...
parse-bbox-func-name=NvDsInferParseYolo
...
```

**NOTE**: The **YOLOv7** resizes the input with center padding. To get better accuracy, use

```bash
[property]
...
maintain-aspect-ratio=1
...
symmetric-padding=1
...
```

## **7. Edit the deepstream_app_config file**

```bash
...
[primary-gie]
...
config-file=/path/to/your/config_infer_primary_yoloV7.txt
```

## **8. Running the model**

```bash
deepstream-app -c deepstream_app_config.txt
```
