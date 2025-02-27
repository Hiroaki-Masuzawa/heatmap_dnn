# このディレクトリ内のファイル
- gen_dataset.ipynb
    - データセット生成用notebook
- pred_objects.ipynb
    - 推論用notebook (ちょっと古いかも)
- pred_puzzle.py
    - データセット推論・評価用コード
- train_puzzle.py
    - 学習コード


# 使い方
## トレーニング 
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 学習の実行
    ```
    cd block_puzzle
    python3 train_puzzle.py --output train_result
    ```

## 推論 
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 推論の実行
    ```
    cd block_puzzle
    python3 pred_puzzle.py --dnnmodel train_result/model_010.pth --input /dataset/puzzle_block/test/image_000001.png --output pred_result.png
    ```
