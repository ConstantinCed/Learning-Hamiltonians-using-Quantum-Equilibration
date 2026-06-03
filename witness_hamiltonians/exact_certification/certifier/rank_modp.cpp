#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

using Entry = std::pair<uint32_t, uint32_t>;
using Row = std::vector<Entry>;

uint32_t mod_pow(uint32_t a, uint32_t e, uint32_t p) {
    uint64_t base = a;
    uint64_t out = 1;
    while (e != 0) {
        if (e & 1U) {
            out = (out * base) % p;
        }
        base = (base * base) % p;
        e >>= 1U;
    }
    return static_cast<uint32_t>(out);
}

template <class T>
void read_exact(std::ifstream &f, T &x) {
    f.read(reinterpret_cast<char *>(&x), sizeof(T));
    if (!f) {
        throw std::runtime_error("unexpected EOF");
    }
}

uint32_t normalize_int(int32_t v, uint32_t p) {
    int64_t x = v;
    x %= static_cast<int64_t>(p);
    if (x < 0) {
        x += p;
    }
    return static_cast<uint32_t>(x);
}

Row normalize_pivot(const std::map<uint32_t, uint32_t> &row, uint32_t inv, uint32_t p) {
    Row out;
    out.reserve(row.size());
    for (const auto &[col, val] : row) {
        uint32_t nv = static_cast<uint32_t>((static_cast<uint64_t>(val) * inv) % p);
        if (nv != 0) {
            out.emplace_back(col, nv);
        }
    }
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "usage: rank_modp MATRIX.bin [target]\n";
        return 2;
    }
    const std::string path = argv[1];
    uint64_t target_override = 0;
    if (argc >= 3) {
        target_override = std::stoull(argv[2]);
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "cannot open " << path << "\n";
        return 2;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "EXR1") {
        std::cerr << "bad matrix magic\n";
        return 2;
    }
    uint64_t nrows = 0;
    uint64_t ncols = 0;
    uint32_t p = 0;
    uint32_t repeated_rows = 0;
    read_exact(f, nrows);
    read_exact(f, ncols);
    read_exact(f, p);
    read_exact(f, repeated_rows);
    (void)repeated_rows;

    std::vector<Row> rows;
    rows.reserve(static_cast<size_t>(nrows));
    uint64_t nnz = 0;
    for (uint64_t r = 0; r < nrows; ++r) {
        uint32_t len = 0;
        read_exact(f, len);
        Row row;
        row.reserve(len);
        for (uint32_t i = 0; i < len; ++i) {
            uint32_t col = 0;
            int32_t val = 0;
            read_exact(f, col);
            read_exact(f, val);
            uint32_t nv = normalize_int(val, p);
            if (nv != 0) {
                row.emplace_back(col, nv);
            }
        }
        nnz += row.size();
        rows.push_back(std::move(row));
    }

    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) {
        if (a.size() != b.size()) {
            return a.size() < b.size();
        }
        if (a.empty() || b.empty()) {
            return a.size() < b.size();
        }
        return a.front().first < b.front().first;
    });

    const uint64_t target = target_override != 0 ? target_override : ncols - 1;
    std::vector<Row> pivots(static_cast<size_t>(ncols));
    uint64_t rank = 0;
    uint64_t processed = 0;
    const auto t0 = std::chrono::steady_clock::now();
    auto last_report = t0;

    for (const Row &source : rows) {
        ++processed;
        std::map<uint32_t, uint32_t> row;
        for (const auto &[col, val] : source) {
            row[col] = val;
        }
        while (!row.empty()) {
            const uint32_t pc = row.begin()->first;
            const uint32_t coeff = row.begin()->second;
            const Row &pivot = pivots[pc];
            if (pivot.empty()) {
                const uint32_t inv = mod_pow(coeff, p - 2, p);
                pivots[pc] = normalize_pivot(row, inv, p);
                ++rank;
                if (rank >= target) {
                    const auto dt = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - t0).count();
                    std::cout << "rank " << rank << " target " << target
                              << " processed_rows " << processed << "/" << nrows
                              << " input_nnz " << nnz << " seconds " << dt << "\n";
                    return 0;
                }
                break;
            }
            for (const auto &[col, pv] : pivot) {
                const uint32_t prod = static_cast<uint32_t>(
                    (static_cast<uint64_t>(coeff) * pv) % p);
                auto it = row.find(col);
                uint32_t cur = it == row.end() ? 0 : it->second;
                uint32_t nv = cur >= prod ? cur - prod : cur + p - prod;
                if (nv == 0) {
                    if (it != row.end()) {
                        row.erase(it);
                    }
                } else if (it == row.end()) {
                    row.emplace(col, nv);
                } else {
                    it->second = nv;
                }
            }
        }

        const auto now = std::chrono::steady_clock::now();
        if (std::chrono::duration<double>(now - last_report).count() > 10.0) {
            last_report = now;
            const auto dt = std::chrono::duration<double>(now - t0).count();
            std::cerr << "progress rows=" << processed << "/" << nrows
                      << " rank=" << rank << "/" << target
                      << " seconds=" << dt << "\n";
        }
    }

    const auto dt = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    std::cout << "rank " << rank << " target " << target
              << " processed_rows " << processed << "/" << nrows
              << " input_nnz " << nnz << " seconds " << dt << "\n";
    return rank >= target ? 0 : 1;
}
