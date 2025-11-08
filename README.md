# Traveling Salesman Problem (TSP) Solver

<div align="center">
  <img src="salesman.jpg" alt="picture" width="50%">
</div>

Proyek ini adalah aplikasi berbasis web untuk menyelesaikan **Traveling Salesman Problem (TSP)** menggunakan berbagai algoritma, seperti:
- **Bruteforce**
- **Greedy**
- **Divide and Conquer**
- **DFS (Depth-First Search)**
- **BFS (Breadth-First Search)**

Aplikasi ini dibangun menggunakan **FastAPI** untuk backend dan **HTML + JavaScript** untuk frontend. Anda dapat memvisualisasikan titik-titik dan rute yang dihasilkan oleh algoritma langsung di browser.

---

## Fitur

1. **Input Titik**:
   - Tambahkan titik secara manual dengan label dan koordinat.
   - Gunakan daftar titik default untuk memulai dengan cepat.
   - Hapus semua titik dengan satu klik.

2. **Algoritma TSP**:
   - Pilih algoritma untuk menyelesaikan TSP.
   - Visualisasi rute yang dihasilkan.
   - Log langkah-langkah algoritma ditampilkan secara deskriptif dalam bahasa Indonesia.

3. **Visualisasi**:
   - Titik-titik dan rute divisualisasikan dalam bentuk grafik interaktif.

---

## Cara Kerja Program

1. **Backend**:
   - Backend menggunakan **FastAPI** untuk menerima permintaan POST dengan titik-titik dan algoritma yang dipilih.
   - Algoritma TSP dijalankan di backend, dan hasilnya dikirim kembali ke frontend.
   - Log langkah-langkah algoritma disediakan untuk membantu memahami cara kerja setiap algoritma.

2. **Frontend**:
   - Frontend memungkinkan pengguna untuk memasukkan titik-titik, memilih algoritma, dan melihat hasilnya.
   - Visualisasi dilakukan menggunakan elemen SVG untuk menggambar titik dan rute.

---

## Instruksi Cara Menjalankan Program

### Prasyarat
Pastikan Anda memiliki:
- **Python 3.8 atau lebih baru** terinstal di sistem Anda.
- **Pip** untuk mengelola dependensi Python.

### Langkah-langkah

1. **Clone Repository**:
   Clone repository ini ke komputer Anda:
   ```bash
   git clone https://github.com/username/tsp-flask-app.git
   cd tsp-flask-app
   ```

2. **Install Dependensi**:
   Install semua dependensi yang diperlukan menggunakan `pip`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan Backend**:
   Jalankan server backend menggunakan `uvicorn`:
   ```bash
   uvicorn backend.app:app --reload --port 8000
   ```

   Backend akan berjalan di `http://localhost:8000`.

4. **Jalankan Frontend**:
   Buka file `index.html` di browser Anda. Anda dapat menggunakan ekstensi **Live Server** di VS Code untuk menjalankan frontend secara lokal.

5. **Gunakan Aplikasi**:
   - Tambahkan titik-titik secara manual atau gunakan daftar default.
   - Pilih algoritma untuk menyelesaikan TSP.
   - Klik tombol "Jalankan Algoritma" untuk melihat hasilnya.

---



