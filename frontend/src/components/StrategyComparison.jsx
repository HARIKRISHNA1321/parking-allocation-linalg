import React, { useState, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { Play, CheckCircle2, TrendingUp, AlertTriangle, ShieldCheck, Scale } from "lucide-react";

export default function StrategyComparison({ onSimulate, simulationData, loading }) {
  const [numVehicles, setNumVehicles] = useState(50);

  const handleRun = () => {
    onSimulate(numVehicles);
  };

  const comp = simulationData?.comparison;

  // Chart data
  const metricsData = comp
    ? [
        {
          metric: "Mean Walk (L1)",
          S1: comp.S1?.mean_walking_distance_l1 || 0,
          S2: comp.S2?.mean_walking_distance_l1 || 0,
          S3: comp.S3?.mean_walking_distance_l1 || 0,
        },
        {
          metric: "Lot Utilization %",
          S1: comp.S1?.lot_utilization_pct || 0,
          S2: comp.S2?.lot_utilization_pct || 0,
          S3: comp.S3?.lot_utilization_pct || 0,
        },
        {
          metric: "Zone Imbalance Std",
          S1: comp.S1?.zone_balance_std || 0,
          S2: comp.S2?.zone_balance_std || 0,
          S3: comp.S3?.zone_balance_std || 0,
        },
      ]
    : [];

  return (
    <div>
      {/* Simulation Controls Card */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
          <div>
            <h2 className="card-title" style={{ margin: 0 }}>
              <Scale size={20} color="#38bdf8" />
              Monte Carlo Strategy Benchmark (Identical Arrival Stream)
            </h2>
            <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginTop: "0.25rem" }}>
              Simulates identical pseudo-random sequence of vehicles across S1, S2, and S3 to evaluate real trade-offs.
            </p>
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <label style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>Vehicles to simulate:</label>
              <input
                type="range"
                min="20"
                max="100"
                step="5"
                value={numVehicles}
                onChange={(e) => setNumVehicles(parseInt(e.target.value))}
                style={{ cursor: "pointer" }}
              />
              <span style={{ fontWeight: "bold", width: "32px", textAlign: "right" }}>{numVehicles}</span>
            </div>

            <button className="btn btn-primary" onClick={handleRun} disabled={loading}>
              <Play size={16} />
              {loading ? "Simulating..." : "Run Simulation"}
            </button>
          </div>
        </div>
      </div>

      {comp && (
        <div>
          {/* Key Metric Comparison Cards */}
          <div className="grid-3col" style={{ marginBottom: "1.5rem" }}>
            {/* S1 Card */}
            <div className="card" style={{ borderTop: "4px solid #3b82f6" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                <span className="badge badge-cyan">STRATEGY S1</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Entry L2 Driven</span>
              </div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: "700", marginBottom: "0.5rem" }}>Nearest to Entry</h3>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.75rem" }}>
                Minimizes vehicle driving distance upon entering lot (&alpha;=0, &beta;=1, &gamma;=0).
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", fontSize: "0.8rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Mean Walk Distance (L1):</span>
                  <strong>{comp.S1.mean_walking_distance_l1} units</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Lot Utilization:</span>
                  <strong style={{ color: "#38bdf8" }}>{comp.S1.lot_utilization_pct}%</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Rejected Vehicles:</span>
                  <strong style={{ color: comp.S1.vehicles_rejected > 0 ? "#f87171" : "#34d399" }}>
                    {comp.S1.vehicles_rejected}
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Zone Imbalance Std:</span>
                  <strong>{comp.S1.zone_balance_std}%</strong>
                </div>
              </div>
            </div>

            {/* S2 Card */}
            <div className="card" style={{ borderTop: "4px solid #10b981" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                <span className="badge badge-emerald">STRATEGY S2</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Destination L1 Driven</span>
              </div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: "700", marginBottom: "0.5rem" }}>Nearest to Destination</h3>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.75rem" }}>
                Minimizes pedestrian walking distance to user's destination (&alpha;=1, &beta;=0, &gamma;=0).
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", fontSize: "0.8rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Mean Walk Distance (L1):</span>
                  <strong style={{ color: "#34d399" }}>{comp.S2.mean_walking_distance_l1} units</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Lot Utilization:</span>
                  <strong style={{ color: "#38bdf8" }}>{comp.S2.lot_utilization_pct}%</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Rejected Vehicles:</span>
                  <strong style={{ color: comp.S2.vehicles_rejected > 0 ? "#f87171" : "#34d399" }}>
                    {comp.S2.vehicles_rejected}
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Zone Imbalance Std:</span>
                  <strong style={{ color: "#fbbf24" }}>{comp.S2.zone_balance_std}%</strong>
                </div>
              </div>
            </div>

            {/* S3 Card */}
            <div className="card" style={{ borderTop: "4px solid #f59e0b" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                <span className="badge badge-amber">STRATEGY S3</span>
                <span style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Global Optimization</span>
              </div>
              <h3 style={{ fontSize: "1.1rem", fontWeight: "700", marginBottom: "0.5rem" }}>Weighted Cost + Prediction</h3>
              <p style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginBottom: "0.75rem" }}>
                Balances walking L1, driving L2, and Stage 7 predicted load ĥ (&alpha;=1, &beta;=0.4, &gamma;=2.5).
              </p>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem", fontSize: "0.8rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Mean Walk Distance (L1):</span>
                  <strong style={{ color: "#fbbf24" }}>{comp.S3.mean_walking_distance_l1} units</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Lot Utilization:</span>
                  <strong style={{ color: "#38bdf8" }}>{comp.S3.lot_utilization_pct}%</strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Rejected Vehicles:</span>
                  <strong style={{ color: comp.S3.vehicles_rejected > 0 ? "#f87171" : "#34d399" }}>
                    {comp.S3.vehicles_rejected}
                  </strong>
                </div>
                <div style={{ display: "flex", justifyContent: "space-between" }}>
                  <span style={{ color: "var(--text-muted)" }}>Zone Imbalance Std:</span>
                  <strong style={{ color: "#34d399" }}>{comp.S3.zone_balance_std}% (Optimal)</strong>
                </div>
              </div>
            </div>
          </div>

          {/* Bar Charts */}
          <div className="grid-2col" style={{ marginBottom: "1.5rem" }}>
            <div className="card">
              <h3 className="card-title">Mean Walking Distance Comparison (Lower is Better)</h3>
              <div style={{ height: 260, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[{ name: "Mean Walk L1", S1: comp.S1.mean_walking_distance_l1, S2: comp.S2.mean_walking_distance_l1, S3: comp.S3.mean_walking_distance_l1 }]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155" }} />
                    <Legend />
                    <Bar dataKey="S1" fill="#3b82f6" name="S1: Entry L2" />
                    <Bar dataKey="S2" fill="#10b981" name="S2: Dest L1" />
                    <Bar dataKey="S3" fill="#f59e0b" name="S3: Weighted + Pred" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="card">
              <h3 className="card-title">Zone Occupancy Imbalance Std Dev (Lower is More Balanced)</h3>
              <div style={{ height: 260, width: "100%" }}>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={[{ name: "Zone Imbalance Std", S1: comp.S1.zone_balance_std, S2: comp.S2.zone_balance_std, S3: comp.S3.zone_balance_std }]}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                    <XAxis dataKey="name" stroke="#94a3b8" />
                    <YAxis stroke="#94a3b8" />
                    <Tooltip contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155" }} />
                    <Legend />
                    <Bar dataKey="S1" fill="#3b82f6" name="S1: Entry L2" />
                    <Bar dataKey="S2" fill="#10b981" name="S2: Dest L1" />
                    <Bar dataKey="S3" fill="#f59e0b" name="S3: Weighted + Pred" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>

          {/* Theoretical Summary Box */}
          <div className="card" style={{ background: "rgba(15, 23, 42, 0.8)", borderLeft: "4px solid #38bdf8" }}>
            <h4 style={{ color: "#38bdf8", marginBottom: "0.5rem", display: "flex", alignItems: "center", gap: "0.5rem" }}>
              <ShieldCheck size={18} />
              Mathematical Findings & Operational Trade-offs
            </h4>
            <p style={{ fontSize: "0.85rem", color: "#e2e8f0", lineHeight: 1.6 }}>
              <strong>Strategy 1 (Nearest Entry):</strong> Drivers park immediately at the perimeter to minimize vehicle maneuvers (L2 norm). However, this creates severe walking penalties for pedestrians destined for the opposite end of the campus or shopping centre.
              <br /><br />
              <strong>Strategy 2 (Nearest Destination):</strong> Minimizes driver walking fatigue (L1 norm) by greedy assignment. While walk distances are minimal initially, it hyper-concentrates vehicles in the destination quadrant, causing local gridlock and spot exhaustion for late arrivals.
              <br /><br />
              <strong>Strategy 3 (Weighted Cost + Stage 7 Prediction):</strong> By infusing the linear autoregressive lag load penalty ĥ into the cost functional C(v, s), S3 prevents quadrant oversaturation, achieving balanced zone occupancy while maintaining low average walking distances.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
