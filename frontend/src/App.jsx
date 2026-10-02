import React, { useState, useEffect } from "react";
import { Layers, Activity, BarChart2, Car, Sparkles, RefreshCw } from "lucide-react";
import LiveLot from "./components/LiveLot";
import MathPipeline from "./components/MathPipeline";
import StrategyComparison from "./components/StrategyComparison";

const API_BASE = "";

export default function App() {
  const [activeTab, setActiveTab] = useState("lot");
  const [lotData, setLotData] = useState(null);
  const [pipelineData, setPipelineData] = useState(null);
  const [simulationData, setSimulationData] = useState(null);
  const [allocationResult, setAllocationResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [allocating, setAllocating] = useState(false);
  const [simulating, setSimulating] = useState(false);
  const [error, setError] = useState(null);

  // Load initial lot and pipeline data
  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [resLot, resPipe] = await Promise.all([
        fetch(`${API_BASE}/api/lot`),
        fetch(`${API_BASE}/api/pipeline`)
      ]);

      if (!resLot.ok || !resPipe.ok) {
        throw new Error("Failed to load initial data from server");
      }

      const dataLot = await resLot.json();
      const dataPipe = await resPipe.json();

      setLotData(dataLot);
      setPipelineData(dataPipe);

      // Pre-load a default simulation
      const resSim = await fetch(`${API_BASE}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_vehicles: 50, strategies: ["S1", "S2", "S3"] })
      });
      if (resSim.ok) {
        const dataSim = await resSim.json();
        setSimulationData(dataSim);
      }
    } catch (err) {
      console.error(err);
      setError(err.message || "Network error communicating with FastAPI backend");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  // Handle vehicle allocation
  const handleAllocate = async (req) => {
    setAllocating(true);
    try {
      const res = await fetch(`${API_BASE}/api/allocate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(req)
      });
      const data = await res.json();
      setAllocationResult(data);

      // Refresh lot state
      const resLot = await fetch(`${API_BASE}/api/lot`);
      if (resLot.ok) {
        const updatedLot = await resLot.json();
        setLotData(updatedLot);
      }
    } catch (err) {
      console.error(err);
      alert("Allocation failed: " + err.message);
    } finally {
      setAllocating(false);
    }
  };

  // Handle lot reset
  const handleReset = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/reset`, { method: "POST" });
      if (res.ok) {
        const data = await res.json();
        setLotData(data.lot);
        setAllocationResult(null);
      }
    } catch (err) {
      console.error(err);
    }
  };

  // Handle simulation trigger
  const handleSimulate = async (nVehicles) => {
    setSimulating(true);
    try {
      const res = await fetch(`${API_BASE}/api/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ n_vehicles: nVehicles, strategies: ["S1", "S2", "S3"] })
      });
      if (res.ok) {
        const data = await res.json();
        setSimulationData(data);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setSimulating(false);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navigation Header */}
      <header className="header">
        <div className="header-content">
          <div className="brand">
            <div className="brand-icon">
              <Car size={22} />
            </div>
            <div>
              <div className="brand-title">Parking Allocation Using Linear Algebra</div>
              <div className="brand-sub">University Mini-Project &bull; UCI Parking Birmingham Grounded</div>
            </div>
          </div>

          <nav className="nav-tabs">
            <button
              className={`nav-btn ${activeTab === "lot" ? "active" : ""}`}
              onClick={() => setActiveTab("lot")}
            >
              <Layers size={16} />
              Page 1: Live Lot
            </button>
            <button
              className={`nav-btn ${activeTab === "pipeline" ? "active" : ""}`}
              onClick={() => setActiveTab("pipeline")}
            >
              <Activity size={16} />
              Page 2: Math Pipeline (10 Stages)
            </button>
            <button
              className={`nav-btn ${activeTab === "strategy" ? "active" : ""}`}
              onClick={() => setActiveTab("strategy")}
            >
              <BarChart2 size={16} />
              Page 3: Strategy Comparison
            </button>
          </nav>
        </div>
      </header>

      {/* Main Body */}
      <main className="main-content">
        {loading ? (
          <div className="card" style={{ textAlign: "center", padding: "4rem" }}>
            <RefreshCw size={36} className="spin-icon" style={{ margin: "0 auto 1rem", color: "#38bdf8" }} />
            <h3 style={{ fontSize: "1.2rem", fontWeight: "600" }}>Loading Linear Algebra Engine & Birmingham Dataset...</h3>
            <p style={{ color: "var(--text-muted)", fontSize: "0.85rem", marginTop: "0.5rem" }}>
              Computing RREF, LU Decomposition, Gram-Schmidt QR, and Eigendecomposition from scratch...
            </p>
          </div>
        ) : error ? (
          <div className="card" style={{ borderLeft: "4px solid #ef4444", padding: "2rem" }}>
            <h3 style={{ color: "#f87171", fontSize: "1.2rem" }}>Error Loading Application</h3>
            <p style={{ color: "var(--text-muted)", margin: "0.5rem 0 1rem" }}>{error}</p>
            <button className="btn btn-primary" onClick={fetchData}>Retry Connection</button>
          </div>
        ) : (
          <>
            {activeTab === "lot" && (
              <LiveLot
                lotData={lotData}
                onAllocate={handleAllocate}
                onReset={handleReset}
                allocationResult={allocationResult}
                loading={allocating}
              />
            )}

            {activeTab === "pipeline" && (
              <MathPipeline pipelineData={pipelineData} />
            )}

            {activeTab === "strategy" && (
              <StrategyComparison
                onSimulate={handleSimulate}
                simulationData={simulationData}
                loading={simulating}
              />
            )}
          </>
        )}
      </main>

      {/* Required University Project Footer */}
      <footer className="footer">
        <p>
          Data: Stolfi (2017), UCI Parking Birmingham, CC BY 4.0 &bull; Linear Algebra implemented from scratch in pure NumPy
        </p>
      </footer>
    </div>
  );
}
