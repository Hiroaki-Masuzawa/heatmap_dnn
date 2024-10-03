# headmap dnn

## 準備
### docker imageの準備
```
cd docker 
./build.sh
```

## datasetの準備
- 入力画像とラベル画像のペアを用意し，その対応をcsv形式で記載する．
- 入力画像とラベル画像は同じ縦横サイズの画像とする．
    - 入力画像はカラー画像
    - ラベル画像は0と255のみで構成されるグレースケール画像
    - 試した解像度は256x256のみだが他の解像度でも問題ないはず．
- csvファイルに記載するパスはプログラムを実行する場所からの相対位置で記載する．
    - csvの例
        ```
        train/image_000000.png,train/label_000000.png 
        train/image_000001.png,train/label_000001.png 
        train/image_000002.png,train/label_000002.png 
        train/image_000003.png,train/label_000003.png 
        ```

## 学習
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 学習スクリプトの実行(例)
    ```
    python train.py --traincsv train.csv
    ```

## 推論
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 推論スクリプトの実行(例)
    ```
    python3 pred.py --input val/image_00000000.png --label val/label_00000000.png --output result_00000000.png 
    ```

## 参考webページ
- pytorch
    - Segmenation ModelのgithubとDocumentation
        - https://github.com/qubvel-org/segmentation_models.pytorch
        - https://smp.readthedocs.io/en/latest/index.html
    - Segmentation Modelの使い方
        - https://qiita.com/tchih11/items/6e143dc639e3454cf577
            - 情報がちょっと古い
        - https://zenn.dev/takiser/articles/35f33b7405a29b
    - pytorchのDataset, Dataloaderの作り方
        - https://qiita.com/mathlive/items/2a512831878b8018db02
    - pytorchの学習ループ
        - https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
- OpenCV
    - OpenCVグレースケールからカラーマップへの変換
        - https://qiita.com/hakoyam/items/e312af5c3b9c9ae58fff