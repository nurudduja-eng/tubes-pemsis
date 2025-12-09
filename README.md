# tubes-pemsis

# 📖 Panduan Supply Chain Optimization

Proyek ini adalah model optimasi rantai pasok multi-tahap yang dikembangkan menggunakan *framework* **Pyomo** (Python Optimization Modeling Objects).

## 🛠️ Prasyarat

Sebelum menjalankan kode, pastikan Anda telah menyiapkan prasyarat berikut:

1.  **Python 3.x:** Terinstal di sistem Anda.
2.  **File Data:** Folder `data/` harus berisi semua file CSV yang dibutuhkan (`lahan.csv`, `penggilingan.csv`, `transport.csv`, dll.).
3.  **Solver CBC:** Anda harus menginstal **CBC Solver** agar Pyomo dapat menyelesaikan model optimasi.

-----

## 📦 Instalasi Library Python

Instal semua *library* Python yang dibutuhkan (Pyomo, Pandas, NumPy):

```bash
pip install pyomo pandas numpy
```

## ⚙️ Instalasi CBC Solver

**Pyomo** adalah alat pemodelan, tetapi ia membutuhkan *solver* eksternal untuk menemukan solusi. Kode kita menggunakan **CBC (Coin-or Branch and Cut)**.

Pilih instruksi instalasi sesuai lingkungan Anda:

### Opsi A: Google Colab / Linux (Direkomendasikan)

Jika Anda menggunakan Google Colab, Anda dapat menginstal CBC langsung melalui terminal:

```bash
# Instal solver CBC
!apt-get install -y -qq coinor-cbc

# Ubah kode Python Anda di fungsi load_cbc_solver()
# agar tidak menggunakan path Windows, tetapi hanya nama solver "cbc"
```

### Opsi B: Windows (Instalasi Mandiri)

1.  **Download CBC:** Kunjungi halaman rilis resmi COIN-OR dan unduh versi CBC yang sesuai untuk sistem Windows (biasanya file `.zip` atau `.msi`).

2.  **Ekstrak/Instal:** Ekstrak file yang diunduh ke lokasi permanen di komputer Anda (contoh: `D:\Solver\cbc`).

3.  **Ubah Path di Kode:** Anda **HARUS** mengubah variabel `CBC_PATH` di awal file `main.py` sesuai dengan lokasi *executable* `cbc.exe` yang baru:

    ```python
    # main.py
    # GANTI PATH INI SESUAI LOKASI CBC.EXE PADA SISTEM BOS ANDA
    CBC_PATH = r"D:\Solver\cbc\bin\cbc.exe" # <-- Ganti path ini!
    ```

### Opsi C: macOS (Menggunakan Homebrew)

Jika Anda menggunakan macOS, Anda dapat menginstal melalui *package manager* Homebrew:

```bash
brew install cbc
```

-----

## ▶️ Cara Menjalankan Kode

Setelah semua library dan CBC Solver terinstal, Anda dapat menjalankan model:

### 1\. **Siapkan Data**

Pastikan folder **`data`** sudah ada dan berisi semua file CSV input.

### 2\. **Jalankan Skrip**

Eksekusi file utama dari terminal:

```bash
python main.py
```

### 3\. **Cek Hasil**

Hasil optimasi, log iterasi, dan ringkasan biaya akan disimpan secara otomatis di dalam folder **`result/`**.

-----

## 📌 Catatan Penting Mengenai Kode

### Variabel Utama yang Perlu Diperhatikan:

  * **`MAX_ITER`**: Jumlah maksimum iterasi untuk metode *re-optimisasi* yang digunakan dalam model (saat mengupdate parameter `H_t`).
  * **`TOL_H`**: Toleransi *threshold* untuk konvergensi iteratif.
  * **`Y_LEVELS`**: Tingkat diskritisasi kualitas yang digunakan dalam model Mixed-Integer Linear Programming (MILP).

### Struktur Model (Pyomo):

Kode ini menggunakan model optimasi yang dipecah menjadi dua bagian utama:

1.  `prepare_params`: Memuat dan memproses semua data CSV menjadi parameter Pyomo.
2.  `build_pyomo_model`: Mendefinisikan **Sets**, **Parameters**, **Variables**, **Constraints**, dan **Objective Function** dari model.
3.  `iterative_solve`: Menjalankan proses solusi berulang (`iterative_solve`) karena adanya *coupling* antara harga (`H_t`) dan kualitas produksi (`K_avg`).