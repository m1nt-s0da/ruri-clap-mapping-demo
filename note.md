ruri-v3 → LAION CLAP マッピングのお試し

データセットは知識蒸留も兼ねて cl-nagoya/ruri-v3-dataset-ft から適当にサンプリング

pos/neg を translategemma:4b (一部はgemma4:e2b) で適当に英語翻訳

projection embedding から l2 normalized dot (cosine sim) で、バッチ内の clap(pos+neg) から pos を選び取るタスク
