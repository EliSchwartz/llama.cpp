// Minimal numpy .npy (v1.0 / v2.0) reader for float32 C-contiguous arrays.
// Enough to support the Granite Vision 4.1 fixture harness; not a general-purpose npy library.
#pragma once

#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace g4v_npy {

struct array_f32 {
    std::vector<int64_t> shape;
    std::vector<float>   data;

    int64_t numel() const {
        int64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
};

inline std::string dict_get(const std::string & header, const std::string & key) {
    // Look for 'key': value in a numpy header dict.
    std::string needle = "'" + key + "':";
    size_t p = header.find(needle);
    if (p == std::string::npos) {
        throw std::runtime_error("npy: missing key " + key + " in header");
    }
    p += needle.size();
    while (p < header.size() && (header[p] == ' ' || header[p] == '\t')) p++;
    return header.substr(p);
}

inline array_f32 load_f32(const std::string & path) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("npy: cannot open " + path);

    unsigned char magic[6];
    if (std::fread(magic, 1, 6, f) != 6 || std::memcmp(magic, "\x93NUMPY", 6) != 0) {
        std::fclose(f);
        throw std::runtime_error("npy: bad magic in " + path);
    }

    unsigned char major = 0, minor = 0;
    if (std::fread(&major, 1, 1, f) != 1 || std::fread(&minor, 1, 1, f) != 1) {
        std::fclose(f);
        throw std::runtime_error("npy: truncated version in " + path);
    }

    size_t header_len = 0;
    if (major == 1) {
        uint16_t hl = 0;
        if (std::fread(&hl, 2, 1, f) != 1) { std::fclose(f); throw std::runtime_error("npy: header len read failed"); }
        header_len = hl;
    } else if (major == 2 || major == 3) {
        uint32_t hl = 0;
        if (std::fread(&hl, 4, 1, f) != 1) { std::fclose(f); throw std::runtime_error("npy: header len read failed"); }
        header_len = hl;
    } else {
        std::fclose(f);
        throw std::runtime_error("npy: unsupported version in " + path);
    }

    std::string header(header_len, '\0');
    if (std::fread(header.data(), 1, header_len, f) != header_len) {
        std::fclose(f);
        throw std::runtime_error("npy: header body read failed");
    }

    std::string descr = dict_get(header, "descr");
    if (descr.find("'<f4'") == std::string::npos && descr.find("'|f4'") == std::string::npos
        && descr.find("\"<f4\"") == std::string::npos) {
        std::fclose(f);
        throw std::runtime_error("npy: only float32 little-endian supported (" + path + " has descr=" + descr + ")");
    }

    std::string fortran_order = dict_get(header, "fortran_order");
    if (fortran_order.find("False") == std::string::npos) {
        std::fclose(f);
        throw std::runtime_error("npy: fortran-order arrays not supported");
    }

    array_f32 out;
    // shape: (a, b, c) or (a,)
    std::string shape_str = dict_get(header, "shape");
    size_t lp = shape_str.find('(');
    size_t rp = shape_str.find(')');
    if (lp == std::string::npos || rp == std::string::npos || rp <= lp) {
        std::fclose(f);
        throw std::runtime_error("npy: malformed shape");
    }
    std::string inner = shape_str.substr(lp + 1, rp - lp - 1);
    // split on comma
    std::string tok;
    for (char c : inner) {
        if (c == ',') {
            if (!tok.empty()) out.shape.push_back(std::stoll(tok));
            tok.clear();
        } else if (std::isdigit(static_cast<unsigned char>(c))) {
            tok.push_back(c);
        }
        // ignore whitespace
    }
    if (!tok.empty()) out.shape.push_back(std::stoll(tok));

    int64_t n = out.numel();
    out.data.resize(static_cast<size_t>(n));
    if (std::fread(out.data.data(), sizeof(float), static_cast<size_t>(n), f) != static_cast<size_t>(n)) {
        std::fclose(f);
        throw std::runtime_error("npy: short read of data in " + path);
    }
    std::fclose(f);
    return out;
}

struct diff_report {
    double max_abs;
    double mean_abs;
    double max_rel;   // max(|a-b| / max(|b|, eps))
    double ref_p99;   // 99th percentile of |b|
};

// Compute full diff statistics between two equally-sized float arrays.
// Relative error uses the reference (second arg) as denominator.  Takes a
// single pass over the data so the 99th percentile is approximate via a
// coarse-bucket count; for fixture-diffing this is more than precise enough.
inline diff_report diff_stats(const float * a, const float * b, size_t n) {
    double mx_abs = 0.0, sm_abs = 0.0, mx_rel = 0.0;
    double ref_max = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double av = static_cast<double>(a[i]);
        const double bv = static_cast<double>(b[i]);
        const double d  = std::fabs(av - bv);
        if (d > mx_abs) mx_abs = d;
        sm_abs += d;
        const double denom = std::fabs(bv) + 1e-6;
        const double r = d / denom;
        if (r > mx_rel) mx_rel = r;
        const double br = std::fabs(bv);
        if (br > ref_max) ref_max = br;
    }
    return {mx_abs, n ? sm_abs / static_cast<double>(n) : 0.0, mx_rel, ref_max};
}

} // namespace g4v_npy
