# ***Deteksi Kerusakan Bearing dengan Menggunakan Pemrosesan Sinyal FFT dan Klasifikasi Machine Learning.***
---
# Domain Proyek
---

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/09f4f7c9-6952-4ab3-98d4-d0e008e5d3bd)

Gambar 1. Bearing Test Rig CWRU

Industri modern mengandalkan mesin dan peralatan bergerak untuk menjalankan berbagai macam operasi. Salah satu komponen kritis dalam mesin-mesin ini adalah bearing (bantalan). Bearing berfungsi untuk mengurangi gesekan antara komponen mesin yang bergerak, sehingga memastikan operasi yang lancar dan efisien. Namun, seiring berjalannya waktu dan penggunaan, bearing dapat mengalami kerusakan yang dapat berdampak serius pada kinerja dan keandalan mesin.

Deteksi dini kerusakan bearing memiliki peran penting dalam menjaga kinerja mesin dan mencegah kegagalan yang tidak terduga. Salah satu metode yang efektif untuk mendeteksi kerusakan pada bearing adalah melalui analisis getaran. Getaran yang dihasilkan oleh bearing yang rusak memiliki karakteristik yang berbeda dari bearing yang sehat. Oleh karena itu, analisis sinyal getaran menjadi sangat penting dalam mendeteksi kerusakan ini.

Pemrosesan Sinyal FFT (Fast Fourier Transform) adalah teknik yang umum digunakan untuk menganalisis data getaran. FFT memungkinkan konversi dari domain waktu ke domain frekuensi, mengungkapkan komponen frekuensi yang ada dalam sinyal. Dengan menerapkan FFT pada data getaran, kita dapat mengidentifikasi frekuensi-frekuensi yang mungkin terkait dengan kerusakan pada bearing.

Untuk meningkatkan akurasi dan efisiensi dalam mengklasifikasikan kerusakan pada bearing, penggunaan Machine Learning menjadi pilihan yang menarik. Teknik-teknik machine learning, seperti klasifikasi, dapat memproses data getaran yang telah diubah ke dalam domain frekuensi (menggunakan FFT) dan mengenali pola-pola yang terkait dengan kerusakan.


# Project Understanding
---
***Problem Statement***

Tujuan Proyek

Tujuan dari penelitian ini adalah mengembangkan metode deteksi dini kerusakan bearing berdasarkan analisis data getaran menggunakan pemrosesan sinyal FFT dan klasifikasi dengan machine learning. Dengan demikian, penelitian ini bertujuan untuk:
1. Menerapkan pemrosesan sinyal FFT pada data getaran untuk mengidentifikasi karakteristik frekuensi terkait dengan kerusakan pada bearing.
2. Menggunakan teknik machine learning, seperti klasifikasi, untuk mengklasifikasikan status bearing (sehat atau rusak) berdasarkan pola-pola frekuensi yang dihasilkan dari analisis FFT.
3. Meningkatkan deteksi dini kerusakan bearing sehingga dapat mengurangi downtime mesin, biaya perbaikan, dan potensi kegagalan yang dapat membahayakan operasi industri.

Manfaat Penelitian

Penelitian ini diharapkan dapat memberikan kontribusi dalam bidang pemeliharaan prediktif dan deteksi dini kerusakan mesin. Dengan menggabungkan pemrosesan sinyal FFT dan teknik machine learning, industri dapat mengoptimalkan penggunaan mesin dan mencegah gangguan operasional yang tidak diinginkan. Selain itu, penelitian ini juga dapat menjadi dasar bagi pengembangan sistem otomatis yang dapat memantau kesehatan mesin secara terus-menerus, berpotensi menghemat waktu dan sumber daya manusia dalam pemeliharaan mesin..

***Solution Statement***
- Pada proyek ini akan dilakukan proses feature engineering dengan menggunakan pemrosesan sinyal getaran menggunakan FFT yang selanjutnya akan dilakukan ekstraksi fitur dengan beberapa parameter statistik seperti Mean, Standard Deviation, Shape Factor, RMS, Peak to Peak, Kurtosis, Skewness, Impulse Factor, Crest Factor, Variance, Clearance Factor, Square Root Amplitude dan Form Factor.
- Pada Proyek ini akan digunakan beberapa model *machine learning* berbasis pohon keputusan yaitu ***Decision Tree***,  ***AdaBoost (Adaptive Boosting)***, ***XGBoost (Extreme Gradient Boosting)***, ***Gradient Boosting***, ***Histogram Boosting***, ***Light Gradient Boosting***, ***Random Forest***, ***CatBoost (Categorical Boosting)***
- Dari ketiga algoritma yang digunakan akan dilakukan perbandingan performa dari setiap model dengan menggunakan evaluasi metrik seperti ***Accuracy***, ***Precision***, ***Recall***, dan ***F1-Score***.


![machine learning workflow drawio (1) drawio](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/049e5b19-7d5d-4b2f-93c8-ee36bb19778a)

Gambar 2. Project Workflow

# Data Understanding
---
Data yang digunakan pada proyek ini adalah data sinyal getaran dalam domain waktu selama bearing drive end motor beroperasi dengan kondisi sebagai berikut:
- Kecepatan Putaran 1797 rpm
- Frekuensi Sampling 48000 Hz 
- Tidak diberi beban 
- Kondisi Bearing yaitu Normal, Outer Fault, Inner Fault, dan Ball Fault
- Terdapat 3 Ukuran Diameter dari Kerusakan Bearing yaitu 0.007 inch, 0.014 inch, 0.021 inch.

Data getaran berasal dari bearing data center CWRU (Case Western Reserve University). Berikut ini link dataset yang berektensi csv yang akan digunakan pada proyek ini -> [**Dataset Bearing CWRU**](https://engineering.case.edu/bearingdatacenter/48k-drive-end-bearing-fault-data)

Pada proyek ini dataset yang digunakan terdiri dari 14 Kondisi yaitu:

Tabel 1. Dataset Bearing
|  Data  |                      Deskripsi                      |
|:------:|:---------------------------------------------------:|
| 7_BA   | Rusak Ball dengan Diameter 0.007 inch               |
| 7_IR   | Rusak Inner dengan Diameter 0.007 inch              |
| 7_OR1  | Rusak Outer dengan Diameter 0.007 inch (Center)     |
| 7_OR2  | Rusak Outer dengan Diameter 0.007 inch (Orthogonal) |
| 7_OR3  | Rusak Outer dengan Diameter 0.007 inch (Opposite)   |
| 14_BA  | Rusak Ball dengan Diameter 0.014 inch               |
| 14_IR  | Rusak Inner dengan Diameter 0.014 inch              |
| 14_OR1 | Rusak Outer dengan Diameter 0.014 inch (Center)     |
| 21_BA  | Rusak Ball dengan Diameter 0.021 inch               |
| 21_IR  | Rusak Inner dengan Diameter 0.021 inch              |
| 21_OR1 | Rusak Outer dengan Diameter 0.021 inch (Center)     |
| 21_OR2 | Rusak Outer dengan Diameter 0.021 inch (Orthogonal) |
| 21_OR3 | Rusak Outer dengan Diameter 0.021 inch (Opposite)   |
| N      | Kondisi Normal Bearing                              |


***Exploratory Data Analysis (EDA)***
1. Jumlah Persebaran Data dari 14 kondisi bearing.
![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/128fab0f-a4bb-443e-9a45-ebaff40d1ccf)

Gambar 3. Jumlah Persebaran Data

Tabel 2. Jumlah Persebaran Data

| Fault  | Jumlah Data |
|--------|-------------|
| 7_BA   | 243         |
| 7_IR   | 243         |
| 7_OR1  | 243         |
| 7_OR2  | 124         |
| 7_OR3  | 129         |
| 14_BA  | 249         |
| 14_IR  | 63          |
| 14_OR1 | 245         |
| 21_BA  | 243         |
| 21_IR  | 244         |
| 21_OR1 | 246         |
| 21_OR2 | 128         |
| 21_OR3 | 130         |
| N      | 243         |

2. Disetiap data tidak ada nilai yang hilang (*missing value*)

# Feature Engineering
--- 
**FFT (Fast Fourier Transform)**

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/80b1248a-35c2-4a83-bc80-58cd65ab30f2)

Gambar 4. Transformasi Domain Waktu menjadi Domain Frekuensi dengan (FFT)

$$\mathcal{F}\{x(t)\} = X(f) = \int_{-\infty}^{\infty} x(t) \, e^{-2\pi i f t} \, dt$$

FFT adalah singkatan dari "Fast Fourier Transform" yang merupakan suatu teknik matematika yang digunakan untuk mengubah sinyal atau data dari domain waktu menjadi domain frekuensi. FFT adalah algoritma yang digunakan untuk melakukan transformasi Fourier dengan efisien, khususnya untuk data berukuran besar. FFT merupakan alat penting dalam analisis dan pengolahan sinyal, karena memungkinkan pemahaman lebih mendalam tentang komponen frekuensi dalam data dan memfasilitasi deteksi pola-pola yang mungkin sulit dilihat dalam domain waktu. Dalam penelitian deteksi kerusakan bearing berdasarkan data getaran, FFT digunakan untuk mengonversi sinyal getaran dari domain waktu menjadi domain frekuensi, sehingga pola frekuensi terkait dengan kerusakan pada bearing dapat diidentifikasi dan dianalisis lebih lanjut.

**Ekstraksi Fitur Statistik**

Ekstraksi fitur statistik adalah proses mengambil informasi penting dari data dengan menerapkan berbagai metrik statistik pada data tersebut. Dalam konteks analisis data, ekstraksi fitur statistik melibatkan perhitungan berbagai statistik deskriptif pada data untuk menggambarkan karakteristiknya. Fitur-fitur statistik ini dapat digunakan untuk menganalisis data, mengenali pola, dan mendukung pengambilan keputusan dalam berbagai aplikasi, termasuk analisis sinyal, pengolahan gambar, dan machine learning.

Ekstraksi fitur statistik memiliki manfaat dalam mengurangi dimensi data, mempertahankan informasi penting, dan membantu mengidentifikasi pola atau anomali dalam data. Dalam aplikasi deteksi kerusakan bearing berdasarkan data getaran, fitur-fitur statistik ini dapat membantu menggambarkan karakteristik getaran yang mungkin berbeda antara bearing yang sehat dan rusak. Fitur-fitur ini kemudian dapat dijadikan input untuk teknik-teknik analisis lebih lanjut, seperti klasifikasi menggunakan machine learning. Integrasi ekstraksi fitur statistik dengan pemrosesan sinyal FFT dapat memberikan wawasan yang lebih komprehensif tentang data getaran dan membantu meningkatkan akurasi deteksi kerusakan pada bearing. Berikut ini parameter statistik yang digunakan pada proyek ini:
1. Mean
   
   Mean adalah Rata-rata dari nilai amplitudo spektrum frekuensi dalam interval waktu tertentu.
   $$Mean = \frac{1}{n} \sum_{i=1}^{n} x_i$$


2. Standard Deviation
   
   Standard Deviation adalah Ukuran seberapa tersebar atau variatif amplitudo spektrum frekuensi dalam interval waktu tertentu.
   $$Standard Deviation = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i-\bar{x})^2}$$

3. Shape Factor
   
   Shape Factor adalah ukuran atau metrik yang menggambarkan bentuk, distribusi, atau karakteristik frekuensi dari sinyal getaran.
   $$Shape Factor = {RMS\over Mean}$$

5. Root Mean Square (RMS)

   RMS adalah ukuran statistik yang memberikan gambaran tentang amplitudo atau besarnya getaran dalam data serta melacak perubahan intensitas getaran pada frekuensi tertentu.

   $$RMS = \sqrt{(\frac{1}{n}) \sum_{i=1}^{n} (x_i)^2}$$

6. Peak to Peak

   Peak to Peak adalah ukuran dari jarak amplitudo antara dua titik puncak dalam sebuah sinyal gelombang. Dalam konteks getaran dan sinyal periodik, ukuran P2P mengukur perbedaan nilai maksimum dan minimum dari sinyal selama periode tertentu.

   $$PeakToPeak = {Max Value - Min Value}$$

7. Kurtosis
   
   Kurtosis adalah Ukuran seberapa tajam atau tumpul puncak distribusi amplitudo spektrum frekuensi dalam interval waktu tertentu.

  $$Kurtosis = {\frac{1}{n} \sum_{i=1}^{N} {(x_i-\bar{x})^4\over \sigma^4}}$$

7. Skewness
   
   Skewness adalah Skewness adalah ukuran statistik yang digunakan untuk mengukur asimetri atau kemiringan distribusi data. Dalam konteks sinyal getaran atau data numerik lainnya, skewness memberikan informasi tentang sejauh mana distribusi data condong ke arah tertentu dari pusat distribusi.

  $$Skewness = {\frac{1}{n} \sum_{i=1}^{N} {(x_i-\bar{x})^3\over \sigma^3}}$$


8. Impluse Factor

   Impulse Factor adalah ukuran yang digunakan dalam analisis getaran untuk mengukur dampak impulsif dalam suatu sinyal getaran. Dampak impulsif mengacu pada perubahan tiba-tiba atau lonjakan tajam dalam amplitudo sinyal, yang dapat mengindikasikan kejadian seperti benturan, tumbukan, atau interaksi mekanis yang cepat.

   $$Impulse Factor = \frac{\sum_{i=1}^{n} |x_i|}{n}$$

9.  Crest Factor

    Crest Factor adalah ukuran yang digunakan dalam analisis getaran atau analisis sinyal lainnya untuk mengukur puncak atau nilai maksimum dari amplitudo sinyal dibandingkan dengan rata-rata amplitudo. Faktor puncak memberikan informasi tentang tingkat kejutan atau intensitas puncak dalam sinyal getaran.

    $$\text{Crest Factor} = \frac{x_{\text{peak}}}{x_{\text{rms}}}$$


10. Variance

    Variance adalah Variance adalah ukuran statistik yang digunakan untuk mengukur sejauh mana set data tersebar atau bervariasi dari nilai rata-rata. Dalam konteks data numerik, variance menggambarkan besarnya variasi atau penyebaran nilai-nilai individu dari nilai rata-rata dalam kumpulan data.

    $$Variance = {\sum_{} (x_i - \bar{x})^2\over n-1}$$


11. Clearance Factor

    Clearance Factor adalah Nilai puncak dibagi dengan nilai rata-rata kuadrat dari akar kuadrat dari amplitudo absolut.

   $$\text{Clearance Factor} = \frac{\text{Max Amplitude}}{\text{Square Root of Amplitude}}$$


12. Square Root Amplitude

    Square Root Amplitude adalah metrik yang digunakan dalam analisis sinyal getaran atau analisis spektral untuk mengukur amplitudo relatif suatu komponen frekuensi dalam suatu sinyal gelombang. Ini melibatkan mengambil akar kuadrat dari amplitudo dalam domain frekuensi untuk mendapatkan besaran amplitudo yang lebih mudah diinterpretasikan.

    $$SQRT = \sqrt{A^2 + B^2}$$


13. Form Factor
    
    Form Factor adalah ukuran statistik yang digunakan dalam analisis sinyal getaran atau analisis gelombang untuk mengukur bentuk atau karakteristik gelombang dalam hubungannya dengan nilai rata-rata dari gelombang tersebut. Form Factor memberikan informasi tentang "tipe" gelombang, apakah lebih mendekati gelombang sinusoidal atau apakah memiliki bentuk yang lebih kompleks.

    $$\text Form Factor  = \frac{\sqrt{\frac{1}{N} \sum_{i=1}^{N} x_i^2}}{\frac{1}{N} \sum_{i=1}^{N} x_i}$$



# Data Preparation
--- 

- Data hasil tahapan Feature Engineering kemudian akan digunakan dalam proses pelatihan model machine learning yang terdiri dari 13 Kolom Fitur (Hasil Ekstraksi) dan 1 Kolom Target (Kondisi Bearing) serta terdiri dari 2773 baris.
- Data Target akan dilakukan proses *label encoder* untuk menggubah nilai kategorik menjadi numerik. 

Tabel 3. Hasil Label Encoder
  
| Kelas  | Nilai Numerik |
|--------|---------------|
| BA_14  | 0             |
| BA_21  | 1             |
| BA_7   | 2             |
| IR_14  | 3             |
| IR_21  | 4             |
| IR_7   | 5             |
| N      | 6             |
| OR1_14 | 7             |
| OR1_21 | 8             |
| OR1_7  | 9             |
| OR2_21 | 10            |
| OR2_7  | 11            |
| OR3_21 | 12            |
| OR3_7  | 13            |

-  *Label encoder* perlu dilakukan karena pada beberapa algoritma *machine learning* membutuhkan input berupa nilai numerik bukan kategori sehingga perlu dilakukan label encoder.
- Inisialisasi Data sebelum dilakukan proses *splitting*, dimana variabel (X) akan mewakili fitur data yang terdiri dari 13 kolom dan variabel (y) mewakili data target (*fault*).
- *Splitting Data*, membagi data menjadi 2 yaitu data *training* (Digunakan untuk melatih model) dan data *testing* (Digunakan untuk memvalidasi hasil dari proses pelatihan model)dengan pembagian 80% data *training* dan 20% *data testing* dengan random_state = 42.
- Jumlah data *training* sebanyak 2218 dan data *testing* sebannyak 555.
- Data akan dibagi menjadi X_train, y_train, y_train, y_test.

# Modelling
--- 

Dalam proyek ini, terdapat delapan model machine learning yang digunakan untuk mengklasifikasikan data sinyal getaran berdasarkan kondisi bearing. Masing-masing model akan menjalani tahap pelatihan dengan menggunakan perintah .fit pada data X_train dan y_train. Hasil pelatihan akan divalidasi menggunakan metode Cross-Validation. Setiap model machine learning akan diinstruksikan untuk melakukan pelatihan dengan menggunakan parameter default yang telah ditetapkan. Berikut ini 8 model *machine learning* yang digunakan pada proyek ini:

***Decision Tree***

*Decision tree* (pohon keputusan) adalah salah satu metode dalam machine learning yang digunakan untuk pengambilan keputusan atau analisis prediktif. Metode ini menggambarkan alur keputusan dan konsekuensi dari berbagai pilihan dengan menggunakan struktur pohon. Setiap simpul dalam pohon mewakili keputusan berdasarkan nilai-nilai dari fitur-fitur yang diberikan, sedangkan cabang-cabang pohon merepresentasikan kemungkinan hasil atau keputusan yang mungkin diambil.

Proses pembentukan decision tree melibatkan pemilihan fitur yang paling penting dan pembagian data ke dalam subset-subset yang lebih kecil berdasarkan nilai-nilai fitur tersebut. Tujuannya adalah untuk meminimalkan ketidakpastian dan memaksimalkan pemisahan kelas target di setiap langkah. Metode ini dapat digunakan untuk masalah klasifikasi (memisahkan data ke dalam kelas atau label yang berbeda) serta regresi (memperkirakan nilai berkelanjutan).

**Kelebihan *decision tree :*** 

- Mudah dipahami dan divisualisasikan.
- Cocok untuk data dengan fitur diskret dan kontinu.
- Mampu menangani data yang kompleks.
  
**Kelemahan *decision tree :*** 

- Rentan terhadap overfitting jika tidak diatur dengan baik.
- Tidak selalu menghasilkan hasil prediksi yang akurat.
- Tidak stabil terhadap perubahan kecil pada data input.


***AdaBoost (Adaptive Boosting)***

*AdaBoost (Adaptive Boosting)* adalah salah satu teknik ensemble learning dalam machine learning yang digunakan untuk meningkatkan kinerja model prediksi dengan menggabungkan beberapa model lemah (weak learners) menjadi satu model yang lebih kuat. Konsep dasar di balik AdaBoost adalah memberikan bobot lebih kepada data yang terjadi kesalahan prediksi pada model sebelumnya, sehingga iteratif memperbaiki kesalahan tersebut.

**Kelebihan *AdaBoost* :**

- Meningkatkan Kinerja: AdaBoost mampu meningkatkan kinerja model dengan menggabungkan beberapa model lemah menjadi satu model ensemble yang kuat, sehingga menghasilkan prediksi yang lebih akurat.
- Mengatasi Overfitting: Dengan memberikan fokus pada data yang sulit diprediksi, AdaBoost cenderung mengurangi kemungkinan overfitting dan meningkatkan kemampuan generalisasi model.
- Interpretabilitas: Meskipun bukan model yang sangat mudah diinterpretasi seperti decision tree tunggal, model ensemble AdaBoost masih memungkinkan untuk dianalisis dan dipahami relatif lebih baik daripada beberapa teknik ensemble lainnya.

Kekurangan AdaBoost:
- Sensitivitas Terhadap Noise dan Outlier: AdaBoost cenderung sensitif terhadap data noise dan outlier, yang dapat mengakibatkan peningkatan bobot pada data yang salah dan mempengaruhi kinerja model.
- Ketergantungan pada Kualitas Data: Performa AdaBoost sangat tergantung pada kualitas data pelatihan. Data yang buruk atau tidak representatif dapat menghasilkan model ensemble yang tidak efektif.
- Memerlukan Penyetelan Parameter: Seperti banyak algoritma machine learning lainnya, AdaBoost memiliki parameter yang harus diatur dengan bijak untuk mendapatkan hasil yang optimal.

***XGBoost (Extreme Gradient Boosting)***

*Extreme Gradient Boosting (XGBoost)* merupakan penerapan gradient boosting yang ditingkatkan. Ia menggunakan teknik regularisasi, optimalisasi berbasis histogram, dan pruning untuk meningkatkan akurasi dan mengurangi overfitting. XGBoost menggabungkan konsep boosting dengan pengoptimalan berbasis gradien (gradient-based optimization) untuk menghasilkan model ensemble yang kuat. Algoritma ini menggunakan fungsi tujuan dan derivatifnya untuk mengarahkan pembelajaran model. Seiring dengan berjalannya iterasi, setiap model lemah (biasanya decision tree) ditambahkan ke dalam ensemble dengan tujuan meminimalkan fungsi tujuan.

**Kelebihan *XGBoost* :** 

- Sangat cepat dan efisien dalam pelatihan, mampu menangani data besar dengan cepat, memiliki banyak fitur tuning, dan menghasilkan model yang sangat akurat.

**Kelemahan *XGBoost* :** 

- Membutuhkan pemahaman yang baik tentang parameter dan tuning yang tepat.
  

***Gradient Boosting***

Gradient Boosting adalah metode ensemble di mana setiap model berikutnya berusaha untuk memperbaiki kesalahan model sebelumnya. Ia melakukan ini dengan menghitung gradien dari fungsi kerugian terhadap prediksi model sebelumnya.

**Kelebihan *Gradient Boosting*:** 

- Menghasilkan model yang kuat, mampu menangani berbagai jenis data dan tugas.
  
**Kekurangan *Gradient Boosting*:** 
- Memerlukan pemahaman yang cukup tentang parameter dan dapat memerlukan waktu pelatihan yang lebih lama terutama pada data yang besar.


***CatBoost (Categorical Boosting)***

CatBoost (Categorical Boosting) adalah sebuah algoritma machine learning yang digunakan untuk tugas-tugas klasifikasi dan regresi. Algoritma ini dikembangkan oleh perusahaan Yandex dan menjadi populer dalam kompetisi data science serta proyek-proyek di industri.

CatBoost adalah bentuk dari algoritma boosting ensemble, yang menggabungkan beberapa model lemah (misalnya, decision trees) menjadi satu model yang kuat. Keunggulan utama CatBoost adalah kemampuannya dalam mengatasi data dengan fitur kategori (data kategorikal) tanpa perlu melakukan preprocessing yang kompleks dengan menggunakan teknik ordered boosting dan menggunakan teknik permutasi acak (random permutation) yang mana pada teknik ini dataset akan berbeda beda pada setiap iterasinya sehingga dapat menghindari overfitting akibat model dilakukan pelatihan menggunakan data yang sama.

**Kelebihan *CatBoost* :**

- Penanganan Fitur Kategorikal yang Baik: CatBoost memiliki kemampuan bawaan untuk menangani fitur kategorikal tanpa perlu pra-pemrosesan yang rumit. Ini mengurangi kerumitan dalam pra-pemrosesan data dan memungkinkan penggunaan fitur-fitur kategorikal langsung dalam model.
- Pemberian Bobot Terhadap Fitur Kategori: CatBoost secara otomatis memberikan bobot yang tepat pada nilai-nilai kategori dalam fitur-fitur, mengurangi risiko overfitting dan meningkatkan kinerja model.
- Regularisasi yang Kuat: CatBoost memiliki berbagai teknik regularisasi seperti regularisasi L2, peringkat, dan kompresi pohon, yang membantu mencegah overfitting dan membuat model lebih umum.
- Kemampuan Penanganan Imbalance Data: CatBoost memiliki fitur untuk mengatasi masalah ketidakseimbangan kelas dalam data, yang sering muncul dalam tugas klasifikasi.


**Kekurangan *CatBoost* :**

- Penggunaan Sumber Daya: Pada beberapa dataset besar atau kompleks, CatBoost dapat memerlukan lebih banyak sumber daya komputasi, seperti waktu pelatihan yang lebih lama dan lebih banyak memori.
- Keterbatasan Terhadap Kategori yang Jarang Muncul: Kategori yang sangat jarang muncul dalam data dapat menjadi masalah, karena CatBoost mungkin mengalami kesulitan dalam mengenali hubungan yang signifikan.
- Pembelajaran Kurang Stabil: Pada beberapa dataset, CatBoost mungkin cenderung lebih sensitif terhadap noise dalam data, yang dapat menyebabkan fluktuasi dalam performa model.


***Histogram Boosting***

Histogram Boosting adalah sebuah metode dalam machine learning yang digunakan untuk meningkatkan performa model prediksi, terutama dalam konteks pohon keputusan atau ensemble learning. Teknik ini sering digunakan dalam algoritma seperti XGBoost dan LightGBM.

Histogram Boosting bekerja dengan memanipulasi histogram distribusi data pada setiap fitur (fitur bin) dalam rangka mempercepat proses pelatihan dan meningkatkan akurasi model. Hal ini dilakukan dengan cara menghitung statistik seperti jumlah label target atau gradien loss di setiap bin fitur. Dengan menggunakan histogram ini, algoritma dapat mengambil keputusan yang lebih cerdas saat membangun pohon keputusan dan memutuskan bagaimana membagi data menjadi subset yang lebih kecil.

**Kelebihan Histogram Boosting:** 
- Lebih cepat daripada metode gradient boosting konvensional, cocok untuk data yang besar.
  
**Kekurangan Histogram Boosting:** 
- Mungkin memerlukan tuning parameter tambahan dan tidak selengkap XGBoost atau LightGBM.


***Light Gradient Boosting***

LGBM (Light Gradient Boosting Machine) adalah sebuah algoritma machine learning yang termasuk dalam keluarga algoritma ensemble boosting. LGBM dirancang khusus untuk meningkatkan efisiensi dan kecepatan dalam pelatihan model, terutama pada dataset yang besar. Algoritma ini berfokus pada histogram-based learning, di mana data dipecah menjadi histogram untuk mempercepat proses pembuatan keputusan dalam membangun pohon keputusan.

Berikut adalah beberapa fitur utama dari LGBM:

- Kecepatan Pelatihan: LGBM menggunakan teknik histogram-based learning untuk membagi data ke dalam bucket (histogram), sehingga mengurangi kompleksitas perhitungan saat membangun pohon. Ini membuatnya lebih cepat dalam pelatihan dibandingkan dengan beberapa algoritma boosting lainnya.
- Penanganan Data Besar: Karena pendekatan histogramnya, LGBM sangat efisien dalam menangani dataset yang besar, bahkan dengan fitur-fitur categorical atau numerik.
- Pengaturan Parameter: LGBM memberikan fleksibilitas dalam pengaturan parameter, yang memungkinkan pengguna untuk mengoptimalkan kinerja model sesuai dengan karakteristik dataset dan kebutuhan bisnis.
- Penanganan Overfitting: Seperti algoritma boosting lainnya, LGBM dapat mencegah overfitting melalui teknik seperti regularisasi dan pruning.
- Keakuratan Tinggi: Dengan pendekatan histogramnya, LGBM cenderung menghasilkan model yang memiliki akurasi prediksi yang tinggi.
- Penanganan Fitur Categorical: LGBM secara alami dapat menangani fitur categorical tanpa perlu melakukan one-hot encoding atau transformasi khusus.


***Random Forest***

Random Forest adalah sebuah algoritma machine learning yang termasuk dalam keluarga algoritma ensemble. Ensemble learning adalah pendekatan di mana beberapa model (biasanya pohon keputusan) digabungkan untuk meningkatkan kinerja prediksi dan mengurangi risiko overfitting.

Random Forest bekerja dengan cara berikut:
- Pembentukan Banyak Pohon Keputusan: Random Forest membentuk sejumlah besar pohon keputusan secara acak. Setiap pohon dibangun dengan mengambil sampel acak dari dataset pelatihan, yang dapat berupa subsampling dengan penggantian (bootstrapping). 
- Pembagian Fitur: Saat membangun setiap pohon, hanya subset acak dari fitur-fitur yang digunakan untuk memutuskan setiap pemisahan node dalam pohon. Ini membantu menghindari model menjadi terlalu korelasi dan mengurangi overfitting.
- Voting atau Averaging: Setelah semua pohon selesai dibangun, Random Forest melakukan agregasi hasil prediksi dari semua pohon. Dalam kasus klasifikasi, ini bisa berarti melakukan voting mayoritas dari kelas yang diprediksi oleh setiap pohon. Dalam kasus regresi, hasil prediksi dari setiap pohon bisa diambil rata-rata atau agregasi lainnya.

Keuntungan dari Random Forest adalah:
- Reduksi Overfitting: Dengan menggabungkan banyak pohon yang dibentuk secara acak, Random Forest dapat mengurangi risiko overfitting yang sering terjadi pada pohon keputusan tunggal.
- Stabilitas dan Kinerja: Random Forest umumnya memiliki kinerja yang stabil dan dapat bekerja dengan baik pada berbagai jenis dataset tanpa perlu penyesuaian parameter yang sangat sensitif.
- Penanganan Fitur: Algoritma ini dapat mengatasi berbagai jenis fitur, termasuk fitur-fitur categorical, numerik, dan missing values.
- Pentingnya Fitur: Random Forest dapat memberikan informasi tentang pentingnya setiap fitur dalam prediksi, yang dapat membantu dalam pemahaman tentang kontribusi fitur-fitur terhadap model.

# Evaluation
--- 
Dalam proyek klasifikasi kondisi bearing ini menggunakan evaluasi metrik seperti ***Accuracy***, ***Precision***, ***Recall***, dan ***F1-Score***.

***Accuracy***

$$Accuracy = {TP + TN\over TP + TN + TN + FP}$$
Mengukur sejauh mana model klasifikasi berhasil memprediksi dengan benar kelas target pada dataset pengujian. 

***Precision***
$$Precision = {TP\over TP + FP}$$
Dalam konteks klasifikasi yang mengukur sejauh mana prediksi positif yang dibuat oleh model adalah benar. Dalam kata lain, precision mengukur proporsi dari benar positif (True Positive) terhadap total prediksi positif (True Positive + False Positive). Metrik ini memberikan gambaran tentang seberapa akurat model dalam mengidentifikasi kelas positif. 

***Recall***
$$Recall = {TP\over TP + FN}$$
Dikenal sebagai sensitivitas atau true positive rate, adalah metrik evaluasi dalam konteks klasifikasi yang mengukur sejauh mana model berhasil mengidentifikasi dan mendeteksi semua kasus positif yang ada dalam dataset. Metrik ini memberikan gambaran tentang seberapa baik model mampu mengenali atau "mengingat" semua contoh positif.

***F1-Score***
$$Recall = {2×precision×recall\over precision + recall}$$
Metrik evaluasi yang menggabungkan precision dan recall menjadi satu angka tunggal untuk memberikan gambaran yang lebih komprehensif tentang kinerja model klasifikasi. Metrik ini berguna untuk mengukur keseimbangan antara kemampuan model dalam mengklasifikasikan dengan benar kelas positif (precision) dan kemampuan model dalam mendeteksi semua contoh positif yang sebenarnya ada (recall).

**Berdasarkan hasil pelatihan didapatkan nilai evaluasi dari setiap model :**

Tabel 4. Hasil Evaluasi Model

| Peringkat | Model                   | Testing Accuracy | Training Accuracy | Precision | Recall    | F1-Score  |
|:---------:|-------------------------|------------------|-------------------|-----------|-----------|-----------|
|     1     | CatBoost                | 96.465541        | 96.212352         | 94.517861 | 93.747602 | 93.697400 |
|     2     | Random Forest           | 96.249455        | 95.175604         | 94.125913 | 92.588675 | 93.070685 |
|     3     | Light Gradient Boosting | 96.249325        | 95.986822         | 94.008108 | 93.455158 | 92.976978 |
|     4     | Histogram Boosting      | 96.141217        | 95.671507         | 94.003470 | 93.448152 | 93.394499 |
|     5     | XGBoost                 | 95.744235        | 95.265288         | 93.930594 | 93.150322 | 93.098054 |
|     6     | Gradient Boosting       | 95.636062        | 95.581823         | 93.056016 | 92.850844 | 93.269058 |
|     7     | Decision Tree           | 94.735161        | 93.913326         | 90.937064 | 91.482720 | 91.748258 |
|     8     | AdaBoost                | 94.482649        | 94.048258         | 91.171282 | 91.052701 | 91.190472 |


Berikut ini hasil komparasi dari beberapa model dengan visualisasi *bar chart* pada setiap metrik evaluasi:

![image](https://github.com/MuhammadNafishZaldinanda/FGA/assets/108967925/9206e592-bfca-483f-b4d1-33050d40ec70)

Gambar 5. Bar Chat Hasil Evaluasi Model

Berdasarkan hasil pelatihan dapat dilihat bahwa nilai evaluasi pada setiap model sudah cukup baik dapat dilihat nilai evaluasi sudah mencapai diatas 90% dengan kondisi tanpa menggunakan parameter optimization dalam proses pembangunan modelnya. Dari hasil pelatihan didapatkan bahwa model *CatBoost* memiliki nilai evaluasi model *testing accuracy* sebesar 96.4% serta training accuracy sebesar 96.2%, dari hasil tersebut bisa dikatakan bahwa model catboost bisa dikategorikan model yang goodfit dilihat dari perbedaan selisih nilai *testing* dan *training accuracy* yang kecil. Dapat disimpulkan bahwa model *CatBoost* adalah model paling efektif dalam mengindetifikasi kondisi dari bearing berdasarkan data sinyal getaran yang sudah dilakukan ekstraksi fitur. Selain itu dari model tree based algorithm yang lain random forest yang menggunakan metode bagging dalam pembangunan modelnya sdah cukup baik dilihat dari hasil evaluasi model yang memiliki perbedaan sedikit dengan catboost yang menggunakan metode boosting dalam proses pembangunan model nya.

# Conclusion
--- 
- Berdasarkan hasil pelatihan ke delapan model sudah cukup baik dilihat dari nilai evaluasi model yang sudah mencapai diatas 90% dengan kondisi tanpa melalui proses parameter optimization atau dalam proses pelatihannya hanya menggunakan parameter default saja.
- Dari hasil pelatihan didapatkan nilai evaluasi model terbaik didapatkan pada model *CatBoost* memiliki nilai evaluasi model paling baik dengan nilai *training accuracy* sebesar 96.4%, *testing accuracy* sebesar 96.2%, *precision* sebesar 94.5%, *recall* sebesar 93.7, *f1-score* sebesar 93.6%. 
- Berdasarkan hasil proyek ini didapatkan bahwa model dengan basis algoritma pohon keputusan terbaik adalah *CatBoost* yang paling optimal dalam mengidentifikasi kerusakan *bearing* dibuktikan dengan kemampuannya untuk melakukan generalisasi pada data dengan baik.
