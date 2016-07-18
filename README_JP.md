# enron_data_analysis
(for a version in english, please check the "README_EN.md")

##プロジェクトの概要
エンロン社の財務データが公式に発表されています。そのデータは１４６社員に対してフィーチャーを４０個？？があり、詐欺の疑いで捕まった社員や政府から申し立てされた社員、起訴を免れるため証言者となった社員も含んでいました。そのデータを生かして詐欺が行った他の社員も明確にするアルゴリズムを開発したプロジェクトとなっています。

使用されたアルゴリズムはSVC, GaussianNB, DecisionTreeClassifier と AdaBoostClassifier, KNeighborsClassifierのなか、最も効果が良かったはDecisionTreeClassifierになりました。 ＊１：当プロジェクトは当時に通っていたオンラインデータサイエンスコース（Udacity – Data Analyst）の一部となりますのでレポートは英語です。プロジェクトのgithubはこちらです。


## ファイルの説明

poi_id.py：
メインファイルとなります。データのクリーニングとアルゴリズムでのフィーティングと構築したモデルがもたらした効果です。

na_count.py：
レポートで必要となった行動のファイルです。具体的にそれぞれのフィーチャーであったNA値を数えた上、グラフをつくるファイルです。

tester.py：
作成したモデルの効果をだす、オンラインコースから用意されたファイルです。

feature_format.py：
フィーチャーデータをpython dictionaryから、sk-learnで使用可能なデータを変更出来るファイル

### 以下のファイルはpoi_id.pyからのエクスポートです。

my_feature_list.pkl：
作成したモデルで使用したフィーチャーリスト

my_dataset.pkl：
作成したモデルで使用したデータ

my_classifier.pkl：
使用したモデル

1st_review_20160203_P05_Marios_Koletsis(final)：
コースで作成したレポート（英語）
