// CUDALM — v0.7 Phase B: authoritative W4A16 production shape census.
//
// Enumerates EVERY real W4A16 GEMV (N, K) projection of a converted
// Qwen3.5-0.8B .cudalm v2 model from THREE sources (reviewer §3 — no
// NCU-grid guessing):
//
//   1. ACTUAL MODEL CONFIG: Qwen35Config loaded from the .cudalm file
//      (WeightFileV2::load — the same path the runtime uses).
//   2. ACTUAL CALL SITES: the projection list mirrors, verbatim per layer
//      type, the int4_gemv_bf16 / batch_int4_gemv_bf16 call sites in
//      src/runtime/qwen35_full_attention.cpp and src/runtime/
//      qwen35_deltanet.cpp (and their weight shapes, which
//      src/runtime/weight_loader_v2.cpp::validate_layer pins):
//        full attention: q_proj (fused [q;gate]), k_proj, v_proj, o_proj,
//                        mlp.gate_proj, mlp.up_proj, mlp.down_proj
//        gated deltanet: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
//                        out_proj, mlp.gate_proj, mlp.up_proj,
//                        mlp.down_proj
//   3. PHASE-A NSYS CSV (optional arg): the batched-only gpukernsum CSV is
//      joined on the rowtile4 grid (= ceil(N/4)): every int4gemv row's
//      instance count MUST be an integer number of traversals for that
//      shape's call rate, and the implied traversal count must be the same
//      across all shapes for a given B class — otherwise the tool aborts
//      (nonzero exit). The census is only "authoritative" when it
//      reconciles with the measurement.
//
// Usage:
//   dump_w4a16_shape_census <full_model.cudalm> [phase_a_batched_nsys_gpukernsum.csv]
//
// Host-only tool (no GPU). Output: the census table on stdout.

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "cudalm/qwen35_config.h"
#include "cudalm/weight_loader_v2.h"

namespace {

struct Projection {
  const char* role;      // weight tensor role (call site)
  int N = 0;             // output rows
  int K = 0;             // input dim
  int layers = 0;        // number of layers with this projection
  int calls_per_trav = 0;  // = layers (one call per layer per traversal)
};

// Mirror of the runtime call sites (see file header). `c` is the config
// loaded from the model file.
std::vector<Projection> census_projections(const cudalm::Qwen35Config& c) {
  int full_layers = 0, lin_layers = 0;
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    if (c.is_full_attention(i)) ++full_layers; else ++lin_layers;
  }

  std::vector<Projection> p;
  auto add = [&](const char* role, int N, int K, int layers) {
    Projection q;
    q.role = role;
    q.N = N;
    q.K = K;
    q.layers = layers;
    q.calls_per_trav = layers;
    p.push_back(q);
  };

  const int H = c.hidden_size;
  const int inter = c.intermediate_size;

  // ---- shared per-layer SwiGLU (both layer types) ------------------------
  // mlp.gate_proj / mlp.up_proj: {intermediate_size, H/2} -> N=inter, K=H
  add("mlp.gate_proj", inter, H, c.num_hidden_layers);
  add("mlp.up_proj", inter, H, c.num_hidden_layers);
  // mlp.down_proj: {H, intermediate_size/2} -> N=H, K=inter
  add("mlp.down_proj", H, inter, c.num_hidden_layers);

  // ---- full-attention-only (full_layers) ----------------------------------
  add("self_attn.q_proj (fused [q;gate])", c.q_proj_out(), H, full_layers);
  add("self_attn.k_proj", c.kv_proj_out(), H, full_layers);
  add("self_attn.v_proj", c.kv_proj_out(), H, full_layers);
  add("self_attn.o_proj", H, c.o_proj_in(), full_layers);

  // ---- gated-deltanet-only (lin_layers) -----------------------------------
  add("linear_attn.in_proj_qkv", c.linear_conv_dim(), H, lin_layers);
  add("linear_attn.in_proj_z", c.linear_value_dim(), H, lin_layers);
  add("linear_attn.in_proj_b", c.lin_num_v_heads, H, lin_layers);
  add("linear_attn.in_proj_a", c.lin_num_v_heads, H, lin_layers);
  add("linear_attn.out_proj", H, c.linear_value_dim(), lin_layers);

  return p;
}

// A census row = one unique (N,K) with its roles and call rate.
struct Row {
  int N, K;
  std::vector<std::string> roles;
  int calls_per_trav = 0;
};

struct CsvCell {
  long b1_inst = 0, b1_ns = 0, b1_have = 0;
  long b2_inst = 0, b2_ns = 0, b2_have = 0;
  long b3_inst = 0, b3_ns = 0, b3_have = 0;
};

// nsys 2022.4 gpukernsum CSV:
// Time (%),Total Time (ns),Instances,Avg (ns),Med (ns),Min (ns),Max (ns),
// StdDev (ns),GridXYZ,BlockXYZ,Name
// GridXYZ is space-separated (" 896    1    1"); Name is the quoted last
// column and may contain commas.
//
// NOTE: the CSV groups by (kernel name, grid) — several census rows can
// SHARE one rowtile4 grid (e.g. N=1024 with K=2048 and K=3584 both give
// grid 256), so cells are joined at GRID granularity, not per (N,K) row.
int parse_csv(const char* path, const std::map<int, int>& grid_rate,
              std::map<int, CsvCell>* csv) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 1;
  }
  std::string line;
  bool hdr_done = false;
  int bad = 0;
  while (std::getline(f, line)) {
    if (!hdr_done) { hdr_done = true; continue; }
    if (line.empty()) continue;
    std::vector<std::string> col;
    {
      std::string cur;
      for (char ch : line) {
        if (ch == ',') { col.push_back(cur); cur.clear(); }
        else cur.push_back(ch);
      }
      col.push_back(cur);
    }
    if (col.size() < 10) continue;
    // Name is the LAST column, quoted, and contains commas: take the text
    // between the first and last double quote on the line.
    const size_t q0 = line.find('"');
    const size_t q1 = line.rfind('"');
    if (q0 == std::string::npos || q1 <= q0) continue;
    const std::string name = line.substr(q0 + 1, q1 - q0 - 1);
    if (name.find("int4gemv_rowtile4_bf16") == std::string::npos) continue;
    const bool is_batch = name.find("batch_int4gemv") != std::string::npos;
    const long inst = std::atol(col[2].c_str());
    const long tot_ns = std::atol(col[1].c_str());
    const std::string& g = col[8];
    // GridXYZ text like " 896    1    1" (space-separated, no parens):
    // first number = x (ceil(N/R)), second = y (B).
    auto nth = [&](int n) {
      long v = 0;
      int idx = 0;  // index of the next number to be closed
      bool in_num = false;
      for (char ch : g) {
        if (std::isdigit(static_cast<unsigned char>(ch))) {
          if (!in_num) { in_num = true; v = 0; }
          v = v * 10 + (ch - '0');
        } else if (in_num) {
          in_num = false;
          if (idx == n) return v;
          ++idx;
        }
      }
      if (in_num && idx == n) return v;
      return 0L;
    };
    const long gx = nth(0);
    const long gy = nth(1);
    if (!grid_rate.count(static_cast<int>(gx))) {
      std::fprintf(stderr,
                   "CSV int4gemv grid x=%ld matches no census shape\n", gx);
      ++bad;
      continue;
    }
    CsvCell& cell = (*csv)[static_cast<int>(gx)];
    if (is_batch) {
      if (gy == 2) { cell.b2_inst += inst; cell.b2_ns += tot_ns; ++cell.b2_have; }
      else if (gy == 3) { cell.b3_inst += inst; cell.b3_ns += tot_ns; ++cell.b3_have; }
      else {
        std::fprintf(stderr, "unexpected batch grid y=%ld for grid x=%ld\n",
                     gy, gx);
        ++bad;
      }
    } else {
      cell.b1_inst += inst;
      cell.b1_ns += tot_ns;
      ++cell.b1_have;
    }
  }
  return bad ? 1 : 0;
}

std::string roles_joined(const Row& r) {
  std::string s;
  for (size_t i = 0; i < r.roles.size(); ++i) {
    if (i) s += " + ";
    s += r.roles[i];
  }
  return s;
}

std::string cell_str(long inst, long ns, long have) {
  if (!have) return "-";
  return std::to_string(inst) + "(" + std::to_string(ns / 1000) + "us)";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: %s <full_model.cudalm> "
                 "[phase_a_batched_nsys_gpukernsum.csv]\n", argv[0]);
    return 2;
  }
  cudalm::WeightFileV2 file;
  cudalm::Status s = cudalm::WeightFileV2::load(argv[1], &file);
  if (!s.ok) {
    std::fprintf(stderr, "failed to load model config from %s: %s\n", argv[1],
                 s.message.c_str());
    return 1;
  }
  const cudalm::Qwen35Config& c = file.config();
  if (!c.valid()) {
    std::fprintf(stderr, "model config in %s is invalid\n", argv[1]);
    return 1;
  }

  const std::vector<Projection> projections = census_projections(c);
  int full_layers = 0, lin_layers = 0;
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    if (c.is_full_attention(i)) ++full_layers; else ++lin_layers;
  }

  // Merge by (N,K) (std::map keeps the table sorted by N then K).
  std::map<std::pair<int, int>, Row> rows;
  for (const Projection& pr : projections) {
    Row& r = rows[std::make_pair(pr.N, pr.K)];
    r.N = pr.N;
    r.K = pr.K;
    r.roles.push_back(pr.role);
    r.calls_per_trav += pr.calls_per_trav;
  }

  long total_calls_per_trav = 0;
  for (const auto& kv : rows) total_calls_per_trav += kv.second.calls_per_trav;

  // Grid-level call rate: rowtile4 grid -> sum of calls/trav over all rows
  // that share the grid (grid-collision group).
  std::map<int, int> grid_rate;
  for (const auto& kv : rows) {
    const int grid = (kv.first.first + 3) / 4;
    grid_rate[grid] += kv.second.calls_per_trav;
  }

  std::map<int, CsvCell> csv;
  const bool have_csv = argc >= 3;
  if (have_csv) {
    if (parse_csv(argv[2], grid_rate, &csv)) return 1;
  }

  std::printf("CUDALM v0.7 Phase B — W4A16 production shape census\n");
  std::printf("model: %s\n", argv[1]);
  std::printf(
      "config: H=%d layers=%d (full_attention=%d, gated_deltanet=%d) "
      "intermediate=%d q_proj_out=%d kv_proj_out=%d o_proj_in=%d "
      "lin_conv_dim=%d lin_value_dim=%d lin_v_heads=%d group_size=%d\n",
      c.hidden_size, c.num_hidden_layers, full_layers, lin_layers,
      c.intermediate_size, c.q_proj_out(), c.kv_proj_out(), c.o_proj_in(),
      c.linear_conv_dim(), c.linear_value_dim(), c.lin_num_v_heads,
      c.group_size);

  std::printf("\n== per-projection list (call-site mirror) ==\n");
  std::printf("%-34s %6s %6s %8s %10s\n", "role", "N", "K", "layers",
              "calls/trav");
  for (const Projection& pr : projections) {
    std::printf("%-34s %6d %6d %8d %10d\n", pr.role, pr.N, pr.K, pr.layers,
                pr.calls_per_trav);
  }
  std::printf("total W4A16 calls per model traversal: %ld\n",
              total_calls_per_trav);

  std::printf("\n== census rows (unique (N,K); the kernel sees (N,K) only) ==\n");
  if (have_csv) {
    std::printf("%-6s %-6s %6s %6s %8s | %-14s %-14s %-14s | %s\n", "N", "K",
                "gridR4", "gridR1", "calls/tr", "B1 inst(ns)", "B2 inst(ns)",
                "B3 inst(ns)", "roles");
  } else {
    std::printf("%-6s %-6s %6s %6s %8s | %s\n", "N", "K", "gridR4", "gridR1",
                "calls/tr", "roles");
  }
  int mismatches = 0;
  std::set<int> printed_grid;
  for (const auto& kv : rows) {
    const Row& r = kv.second;
    const int grid4 = (r.N + 3) / 4;
    const int grid1 = r.N;
    const CsvCell cell = csv.count(grid4) ? csv[grid4] : CsvCell{};
    std::printf("%-6d %-6d %6d %6d %8d | ", r.N, r.K, grid4, grid1,
                r.calls_per_trav);
    if (have_csv) {
      // grid_rate[grid4] > r.calls_per_trav means the grid is shared by
      // multiple census rows (the CSV groups them into one row); the cell
      // is then printed only on the FIRST row of the group.
      const bool shared = grid_rate[grid4] > r.calls_per_trav;
      const bool first_of_group = !shared || !printed_grid.count(grid4);
      if (shared) printed_grid.insert(grid4);
      if (first_of_group) {
        std::printf("%-14s %-14s %-14s | %s%s\n",
                    cell_str(cell.b1_inst, cell.b1_ns, cell.b1_have).c_str(),
                    cell_str(cell.b2_inst, cell.b2_ns, cell.b2_have).c_str(),
                    cell_str(cell.b3_inst, cell.b3_ns, cell.b3_have).c_str(),
                    shared
                        ? ("[grid " + std::to_string(grid4) + " group, " +
                           std::to_string(grid_rate[grid4]) +
                           " calls/tr total: ").c_str()
                        : "",
                    roles_joined(r).c_str());
      } else {
        std::printf("%-14s %-14s %-14s | (grid %d group above) %s\n", "-",
                    "-", "-", grid4, roles_joined(r).c_str());
      }
    } else {
      std::printf("%s\n", roles_joined(r).c_str());
    }
  }
  if (have_csv) {
    // Reconciliation at GRID granularity:
    //   inst_grid == (sum of calls/trav over the grid group) x T,
    //   with the same implied T across all grids for a given B class.
    auto implied = [&](int cls) -> long {
      long ref = -1;
      for (const auto& gv : grid_rate) {
        const CsvCell cell = csv.count(gv.first) ? csv[gv.first] : CsvCell{};
        long inst = 0, have = 0;
        if (cls == 1) { inst = cell.b1_inst; have = cell.b1_have; }
        if (cls == 2) { inst = cell.b2_inst; have = cell.b2_have; }
        if (cls == 3) { inst = cell.b3_inst; have = cell.b3_have; }
        if (!have) continue;
        if (inst % gv.second != 0) {
          std::fprintf(stderr,
                       "CENSUS MISMATCH: grid %d B%d inst %ld not a multiple "
                       "of grid calls/trav %d\n",
                       gv.first, cls, inst, gv.second);
          ++mismatches;
          continue;
        }
        const long t = inst / gv.second;
        if (ref < 0) ref = t;
        else if (t != ref) {
          std::fprintf(stderr,
                       "CENSUS MISMATCH: B%d implied traversal count differs "
                       "across grids (%ld vs %ld)\n",
                       cls, ref, t);
          ++mismatches;
        }
      }
      return ref;
    };
    const long t1 = implied(1), t2 = implied(2), t3 = implied(3);
    std::printf("implied traversals over the whole profiled process: "
                "B1=%ld B2=%ld B3=%ld (sum=%ld)\n",
                t1, t2, t3, t1 + t2 + t3);
  }

  if (mismatches) {
    std::fprintf(stderr, "census FAILED: %d mismatches\n", mismatches);
    return 1;
  }
  if (have_csv) std::printf("census reconciles with Phase A CSV: OK\n");
  return 0;
}
