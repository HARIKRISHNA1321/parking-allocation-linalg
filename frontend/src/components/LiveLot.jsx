import React, { useState } from "react";
import { Car, Zap, Accessibility, Compass, RotateCcw, CheckCircle2, AlertCircle, ArrowRight, Layers } from "lucide-react";

export default function LiveLot({ lotData, onAllocate, onReset, allocationResult, loading }) {
  const [vehicleType, setVehicleType] = useState("standard");
  const [destination, setDestination] = useState("Market");
  const [strategy, setStrategy] = useState("S3");

  if (!lotData) {
    return <div className="card" style={{ textAlign: "center", padding: "3rem" }}>Loading Parking Lot Grid...</div>;
  }

  const { spots, dimensions, entrance, destinations, zones, stats, model_disclosure } = lotData;

  const handleSubmit = (e) => {
    e.preventDefault();
    onAllocate({
      vehicle_type: vehicleType,
      destination: destination,
      strategy: strategy
    });
  };

  const allocatedSpot = allocationResult?.allocated_spot;
  const alternatives = allocationResult?.alternatives || [];
  const norms = allocatedSpot?.norm_comparison_to_dest;

  const getSpotIcon = (type) => {
    if (type === "ev") return <Zap size={11} color="#f59e0b" />;
    if (type === "accessible") return <Accessibility size={11} color="#38bdf8" />;
    return null;
  };

  const getZoneColor = (zoneName) => {
    if (zoneName === "Zone A") return "rgba(59, 130, 246, 0.25)";
    if (zoneName === "Zone B") return "rgba(16, 185, 129, 0.25)";
    if (zoneName === "Zone C") return "rgba(245, 158, 11, 0.25)";
    return "rgba(139, 92, 246, 0.25)";
  };

  return (
    <div>
      {/* Real vs Model Disclosure Banner */}
      <div className="disclosure-banner">
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span className="badge-real">Real Data Grounding</span>
          <span>{model_disclosure.real_aspect}</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
          <span className="badge-model">Our Model</span>
          <span>{model_disclosure.modelled_aspect}</span>
        </div>
      </div>

      {/* Grid and Controls Container */}
      <div className="grid-2col" style={{ alignItems: "start" }}>
        
        {/* Left Column: Interactive 10x12 Lot Visualizer */}
        <div className="card">
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
            <h2 className="card-title" style={{ margin: 0 }}>
              <Layers size={20} color="#38bdf8" />
              10x12 Lot Grid Matrix L (120 Spots)
            </h2>
            <button className="btn btn-secondary" onClick={onReset} title="Reset Lot State" style={{ padding: "0.35rem 0.75rem", fontSize: "0.75rem" }}>
              <RotateCcw size={14} />
              Reset Lot
            </button>
          </div>

          {/* Stats Bar */}
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap", fontSize: "0.8rem" }}>
            <span style={{ color: "#34d399" }}>● Free: {stats.free}</span>
            <span style={{ color: "#f87171" }}>● Occupied: {stats.occupied}</span>
            <span style={{ color: "#94a3b8" }}>■ Blocked: {stats.blocked}</span>
            <span style={{ color: "#38bdf8", fontWeight: "bold" }}>Utilization: {stats.utilization_pct}%</span>
          </div>

          {/* Lot Grid */}
          <div className="lot-grid-container">
            <div className="lot-grid">
              {spots.map((spot) => {
                const isAllocated = allocatedSpot && allocatedSpot.spot_id === spot.spot_id;
                let statusClass = "cell-free";
                if (spot.status === 1) statusClass = "cell-occupied";
                if (spot.status === 2) statusClass = "cell-blocked";
                if (isAllocated) statusClass = "cell-selected";

                return (
                  <div
                    key={spot.spot_id}
                    className={`lot-cell ${statusClass}`}
                    style={{
                      backgroundColor: isAllocated ? undefined : (spot.status === 0 ? getZoneColor(spot.zone) : undefined)
                    }}
                    title={`Spot #${spot.spot_id} (${spot.row}, ${spot.col})\nZone: ${spot.zone}\nType: ${spot.type}\nStatus: ${spot.status_label}`}
                  >
                    <span style={{ fontSize: "0.6rem" }}>{spot.spot_id}</span>
                    <div>{getSpotIcon(spot.type)}</div>
                    <span className="zone-tag">{spot.zone.slice(-1)}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Zones Legend */}
          <div style={{ marginTop: "1rem", display: "grid", gridTemplateColumns: "repeat(2, 1fr)", gap: "0.5rem" }}>
            {Object.entries(zones).map(([zk, info]) => (
              <div key={zk} className="zone-pill" style={{ background: info.color + "22", border: `1px solid ${info.color}55` }}>
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: info.color }}></span>
                <span><strong>{zk}:</strong> {info.name} ({info.spots} spots, Real Cap: {info.real_capacity})</span>
              </div>
            ))}
          </div>

          {/* Key landmarks */}
          <div style={{ marginTop: "0.75rem", fontSize: "0.75rem", color: "var(--text-muted)", display: "flex", gap: "1rem", flexWrap: "wrap" }}>
            <span>🚪 <strong>Entrance:</strong> ({entrance[0]}, {entrance[1]})</span>
            <span>🏛️ <strong>Town Hall:</strong> ({destinations["Town Hall"].coord.join(",")})</span>
            <span>🛒 <strong>Market:</strong> ({destinations["Market"].coord.join(",")})</span>
            <span>🛍️ <strong>Mailbox:</strong> ({destinations["Mailbox Mall"].coord.join(",")})</span>
          </div>
        </div>

        {/* Right Column: Park a Vehicle & Cost Breakdown */}
        <div>
          {/* Form */}
          <div className="card" style={{ marginBottom: "1.25rem" }}>
            <h2 className="card-title">
              <Car size={20} color="#38bdf8" />
              Park an Arriving Vehicle
            </h2>
            <form onSubmit={handleSubmit}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                <div className="form-group">
                  <label className="form-label">Vehicle Type</label>
                  <select className="form-select" value={vehicleType} onChange={(e) => setVehicleType(e.target.value)}>
                    <option value="standard">Standard Passenger Car</option>
                    <option value="compact">Compact / City Mini</option>
                    <option value="ev">Electric Vehicle (Requires Charger)</option>
                    <option value="accessible">Accessible / Disabled Permit</option>
                  </select>
                </div>

                <div className="form-group">
                  <label className="form-label">Driver Destination</label>
                  <select className="form-select" value={destination} onChange={(e) => setDestination(e.target.value)}>
                    {Object.keys(destinations).map((k) => (
                      <option key={k} value={k}>{destinations[k].name}</option>
                    ))}
                  </select>
                </div>
              </div>

              <div className="form-group">
                <label className="form-label">Allocation Strategy (Cost Functional)</label>
                <select className="form-select" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                  <option value="S3">S3: Full Weighted Cost + Prediction (alpha=1.0, beta=0.4, gamma=2.5)</option>
                  <option value="S1">S1: Nearest to Entrance (Min Driving L2, alpha=0, beta=1, gamma=0)</option>
                  <option value="S2">S2: Nearest to Destination (Min Walking L1, alpha=1, beta=0, gamma=0)</option>
                </select>
              </div>

              <button type="submit" className="btn btn-primary" style={{ width: "100%" }} disabled={loading}>
                {loading ? "Computing Linear Algebra Argmin..." : "Allocate Optimal Spot (argmin C)"}
              </button>
            </form>
          </div>

          {/* Allocation Results */}
          {allocationResult && (
            <div>
              {allocationResult.success ? (
                <div className="card" style={{ borderLeft: "4px solid #facc15" }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                    <div>
                      <span className="badge badge-amber" style={{ marginBottom: "0.3rem" }}>OPTIMAL SPOT ASSIGNED</span>
                      <h3 style={{ fontSize: "1.4rem", fontWeight: "800", color: "#fef08a" }}>
                        Spot #{allocatedSpot.spot_id} &nbsp;
                        <span style={{ fontSize: "0.9rem", color: "var(--text-muted)", fontWeight: "normal" }}>
                          (Row {allocatedSpot.row}, Col {allocatedSpot.col}) &bull; {allocatedSpot.zone}
                        </span>
                      </h3>
                    </div>
                    <div style={{ textAlign: "right" }}>
                      <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Total Minimized Cost</div>
                      <div style={{ fontSize: "1.3rem", fontWeight: "800", color: "#38bdf8" }}>{allocatedSpot.total_cost}</div>
                    </div>
                  </div>

                  {/* Cost Breakdown */}
                  <div style={{ background: "rgba(15, 23, 42, 0.7)", padding: "0.75rem", borderRadius: "8px", marginBottom: "0.75rem" }}>
                    <div style={{ fontSize: "0.75rem", fontWeight: "bold", color: "var(--text-muted)", marginBottom: "0.5rem" }}>
                      COST BREAKDOWN: C(v, s) = &alpha;&bull;||p-d||₁ + &beta;&bull;||p-e||₂ + &gamma;&bull;ĥ[zone]
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem", fontSize: "0.8rem" }}>
                      <div style={{ borderLeft: "2px solid #38bdf8", paddingLeft: "0.4rem" }}>
                        <div style={{ color: "var(--text-muted)" }}>Walk (L₁ Manhattan)</div>
                        <div style={{ fontWeight: "700" }}>{allocatedSpot.breakdown.walk_distance_l1} grid units</div>
                        <div style={{ fontSize: "0.7rem", color: "#38bdf8" }}>Cost term: +{allocatedSpot.breakdown.walk_cost_term}</div>
                      </div>
                      <div style={{ borderLeft: "2px solid #10b981", paddingLeft: "0.4rem" }}>
                        <div style={{ color: "var(--text-muted)" }}>Drive (L₂ Euclidean)</div>
                        <div style={{ fontWeight: "700" }}>{allocatedSpot.breakdown.drive_distance_l2} units</div>
                        <div style={{ fontSize: "0.7rem", color: "#10b981" }}>Cost term: +{allocatedSpot.breakdown.drive_cost_term}</div>
                      </div>
                      <div style={{ borderLeft: "2px solid #f59e0b", paddingLeft: "0.4rem" }}>
                        <div style={{ color: "var(--text-muted)" }}>Zone Forecast (ĥ)</div>
                        <div style={{ fontWeight: "700" }}>{Math.round(allocatedSpot.breakdown.predicted_zone_load * 100)}% load</div>
                        <div style={{ fontSize: "0.7rem", color: "#f59e0b" }}>Cost term: +{allocatedSpot.breakdown.congestion_cost_term}</div>
                      </div>
                    </div>
                  </div>

                  {/* Norm Comparison Widget */}
                  {norms && (
                    <div style={{ background: "rgba(59, 130, 246, 0.08)", padding: "0.6rem 0.75rem", borderRadius: "8px", marginBottom: "0.75rem", fontSize: "0.75rem" }}>
                      <strong>Vector Norm Comparison to Target Destination:</strong>
                      <div style={{ display: "flex", gap: "1rem", marginTop: "0.3rem" }}>
                        <span><strong>L₁ (Manhattan):</strong> {norms.l1_manhattan}</span>
                        <span><strong>L₂ (Euclidean):</strong> {norms.l2_euclidean.toFixed(2)}</span>
                        <span><strong>L∞ (Chebyshev):</strong> {norms.linf_chebyshev}</span>
                      </div>
                    </div>
                  )}

                  {/* Top 3 Alternatives */}
                  {alternatives.length > 0 && (
                    <div>
                      <div style={{ fontSize: "0.75rem", fontWeight: "bold", color: "var(--text-muted)", marginBottom: "0.35rem" }}>
                        TOP RUNNER-UP ALTERNATIVES:
                      </div>
                      <div style={{ display: "flex", gap: "0.5rem" }}>
                        {alternatives.map((alt, idx) => (
                          <div key={alt.spot_id} style={{ flex: 1, background: "rgba(30, 41, 59, 0.8)", padding: "0.4rem 0.6rem", borderRadius: "6px", fontSize: "0.75rem" }}>
                            <div><strong>#{alt.spot_id}</strong> ({alt.zone})</div>
                            <div style={{ color: "var(--text-muted)" }}>Cost: {alt.cost} <span style={{ color: "#f87171" }}>(+{alt.cost_diff})</span></div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              ) : (
                <div className="card" style={{ borderLeft: "4px solid #ef4444" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", color: "#f87171" }}>
                    <AlertCircle size={20} />
                    <strong>Allocation Failed</strong>
                  </div>
                  <p style={{ marginTop: "0.5rem", fontSize: "0.85rem", color: "var(--text-muted)" }}>{allocationResult.message}</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
