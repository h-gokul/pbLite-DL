For Phase 1:

To find edges on image number 7,
```
python3 Wrapper.py --ImageName 7.jpg
```

For Phase 2,

For the Simple First Neural network, run:
Training:
```
    python3 Train.py --BasePath ../CIFAR10/Train/ --CheckPointPath ../Checkpoints/BasicModel/ --NumEpochs 25 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath Logs/BasicModel/ --ModelName BasicModel 
```
Testing:
```
    python3 Test.py --ModelName BasicModel --ModelPath ../Checkpoints/BasicModel/24model.ckpt --BasePath ../CIFAR10/Test/ --LabelsPath ./TxtFiles/LabelsTest.txt 
```

For the Modified network, run:
Training:
```
python3 Train.py --BasePath ../CIFAR10/Train/ --CheckPointPath ../Checkpoints/BasicModel2/ --NumEpochs 50 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath Logs/BasicModel2/ --ModelName BasicModel2 
```
Testing:
```
python3 Test.py --ModelName BasicModel2 --ModelPath ../Checkpoints/BasicModel2/49model.ckpt --BasePath ../CIFAR10/Test/ --LabelsPath ./TxtFiles/LabelsTest.txt  
```

For Simple ResNet, run:
Training:
```
    python3 Train.py --BasePath ../CIFAR10/Train/ --CheckPointPath ../Checkpoints/ResNet2/ --NumEpochs 50 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath Logs/ResNet2/ --ModelName ResNet2 
```
Testing:
```
    python3 Test.py --ModelName ResNet2  --ModelPath ../Checkpoints/ResNet2/49model.ckpt --BasePath ../CIFAR10/Test/ --LabelsPath ./TxtFiles/LabelsTest.txt 
```

For ResNet pooling version, run:
Training:
```
   python3 Train.py --BasePath ../CIFAR10/Train/ --CheckPointPath ../Checkpoints/ResNet/ --NumEpochs 50 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath Logs/ResNet/ --ModelName ResNet 
```

Testing:
```
python3 Test.py --ModelName ResNet --ModelPath ../Checkpoints/ResNet/49model.ckpt --BasePath ../CIFAR10/Test/ --LabelsPath ./TxtFiles/LabelsTest.txt 
```

For DenseNet, run:
Training:
```
python3 Train.py --BasePath ../CIFAR10/Train/ --CheckPointPath ../Checkpoints/DenseNet/ --NumEpochs 50 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath Logs/DenseNet/ --ModelName DenseNet 
```
Testing:
```
python3 Test.py --ModelName DenseNet --ModelPath ../Checkpoints/DenseNet/49model.ckpt --BasePath ../CIFAR10/Test/ --LabelsPath ./TxtFiles/LabelsTest.txt  
```

For Resnext, run:
Training:
```
python3 Train.py --BasePath ../CIFAR10/Train/ --CheckPointPath ../Checkpoints/ResNext/ --NumEpochs 25 --DivTrain 1 --MiniBatchSize 64 --LoadCheckPoint 0 --LogsPath Logs/ResNext/ --ModelName ResNext 
```
Testing:
```
python3 Test.py --ModelName ResNext --ModelPath ../Checkpoints/ResNext/24model.ckpt --BasePath ../CIFAR10/Test/ --LabelsPath ./TxtFiles/LabelsTest.txt 
```


 


 
