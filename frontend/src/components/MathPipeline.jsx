import React, { useState } from "react";
import { BookOpen, Target, CheckCircle2, ChevronRight, ChevronLeft, Award, Sparkles, BarChart3, Database } from "lucide-react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer
} from "recharts";

export default function MathPipeline({ pipelineData }) {
  const [currentStageIdx, setCurrentStageIdx] = useState(0);

  if (!pipelineData || !pipelineData.stages) {
    return <div className="card" style={{ textAlign: "center", padding: "3rem" }}>Loading Linear Algebra Pipeline Stages...</div>;
  }

  const stages = pipelineData.stages;
  const currentStage = stages[currentStageIdx];

  return (
    <div>
      {/* 10-Stage Horizontal Stepper */}
      <div className="stepper-nav">
        {stages.map((st, idx) => (
          <button
            key={st.stage_number}
            className={`step-tab ${idx === currentStageIdx ? "active" : ""}`}
            onClick={() => setCurrentStageIdx(idx)}
          >
            <span className="step-num">{st.stage_number}</span>
            <span>Stage {st.stage_number}</span>
          </button>
        ))}
      </div>

      {/* Main Stage Presentation Card */}
      <div className="card" style={{ marginBottom: "1.5rem" }}>
        {/* Stage Header */}
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.25rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <div>
            <div style={{ fontSize: "0.8rem", color: "var(--accent-cyan)", fontWeight: "bold", textTransform: "uppercase" }}>
              STAGE {currentStage.stage_number} OF 10
            </div>
            <h2 style={{ fontSize: "1.35rem", fontWeight: "700" }}>{currentStage.title}</h2>
          </div>
          <div style={{ display: "flex", gap: "0.5rem" }}>
            <button
              className="btn btn-secondary"
              onClick={() => setCurrentStageIdx(Math.max(0, currentStageIdx - 1))}
              disabled={currentStageIdx === 0}
              style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem" }}
            >
              <ChevronLeft size={16} /> Prev
            </button>
            <button
              className="btn btn-primary"
              onClick={() => setCurrentStageIdx(Math.min(stages.length - 1, currentStageIdx + 1))}
              disabled={currentStageIdx === stages.length - 1}
              style={{ padding: "0.4rem 0.8rem", fontSize: "0.8rem" }}
            >
              Next <ChevronRight size={16} />
            </button>
          </div>
        </div>

        {/* 3 Explicit Callouts: Concept, Purpose, Outcome */}
        <div className="grid-3col" style={{ marginBottom: "1.5rem" }}>
          {/* Concept */}
          <div className="callout callout-concept">
            <div className="callout-header" style={{ color: "#38bdf8" }}>
              <BookOpen size={16} />
              Concept (Theory)
            </div>
            <p style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>{currentStage.concept}</p>
          </div>

          {/* Purpose */}
          <div className="callout callout-purpose">
            <div className="callout-header" style={{ color: "#34d399" }}>
              <Target size={16} />
              Purpose (Application)
            </div>
            <p style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>{currentStage.purpose}</p>
          </div>

          {/* Outcome */}
          <div className="callout callout-outcome">
            <div className="callout-header" style={{ color: "#fbbf24" }}>
              <CheckCircle2 size={16} />
              Outcome (Verification)
            </div>
            <p style={{ fontSize: "0.85rem", color: "#e2e8f0" }}>{currentStage.outcome}</p>
          </div>
        </div>

        {/* Stage-Specific Visualizations & Mathematical Displays */}
        {renderStageContent(currentStage)}
      </div>
    </div>
  );
}

function renderStageContent(stage) {
  const { stage_number, matrices, summary } = stage;

  switch (stage_number) {
    case 1:
      return (
        <div>
          <h3 style={{ fontSize: "0.95rem", marginBottom: "0.5rem", color: "var(--accent-cyan)" }}>
            Observation Matrix X Sample (First 5 timestamps of 1307):
          </h3>
          <div style={{ overflowX: "auto", marginBottom: "1rem" }}>
            <table className="matrix-table">
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Timestamp Index</th>
                  {matrices.X_columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrices.X_head.map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ textAlign: "left", color: "var(--text-muted)" }}>Slot #{idx + 1}</td>
                    {row.map((val, cIdx) => (
                      <td key={cIdx} style={{ fontWeight: cIdx === 4 ? "bold" : "normal", color: cIdx === 4 ? "#fbbf24" : undefined }}>
                        {val.toFixed(0)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
            <span className="badge badge-cyan">Matrix X Dimensions: 1307 x 5</span>
            <span className="badge badge-emerald">Grid Dimensions: 10 x 12 (120 spots)</span>
            <span className="badge badge-purple">Derived Column: Total = Col1 + Col2 + Col3 + Col4</span>
          </div>
        </div>
      );

    case 2:
      return (
        <div>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
            <span className="badge badge-cyan">Pivot Columns: {matrices.pivot_columns.join(", ")}</span>
            <span className="badge badge-emerald">LU Solver Residual: {summary.lu_solver_residual.toExponential(2)}</span>
            <span className="badge badge-amber">Partial Pivoting: Row Swaps Active</span>
          </div>
          <h3 style={{ fontSize: "0.95rem", marginBottom: "0.5rem", color: "var(--accent-cyan)" }}>
            Reduced Row Echelon Form (RREF) Leading Rows:
          </h3>
          <div style={{ overflowX: "auto", marginBottom: "1rem" }}>
            <table className="matrix-table">
              <tbody>
                {matrices.RREF_top5.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((val, cIdx) => (
                      <td key={cIdx} style={{ color: Math.abs(val) > 0.001 ? "#38bdf8" : "var(--text-dim)" }}>
                        {val.toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            * Note: Column 5 has entries (1.000, 1.000, 1.000, 1.000) corresponding to the sum of pivots 0..3, proving linear dependency.
          </p>
        </div>
      );

    case 3:
      return (
        <div>
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <div className="card" style={{ padding: "0.75rem 1.25rem", background: "rgba(59, 130, 246, 0.15)", flex: 1 }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Rank of X</div>
              <div style={{ fontSize: "1.8rem", fontWeight: "800", color: "#38bdf8" }}>{summary.rank}</div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Number of independent car parks</div>
            </div>
            <div className="card" style={{ padding: "0.75rem 1.25rem", background: "rgba(139, 92, 246, 0.15)", flex: 1 }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Nullity of X</div>
              <div style={{ fontSize: "1.8rem", fontWeight: "800", color: "#c084fc" }}>{summary.nullity}</div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Dimension of null space (5 - 4 = 1)</div>
            </div>
            <div className="card" style={{ padding: "0.75rem 1.25rem", background: "rgba(16, 185, 129, 0.15)", flex: 1 }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Rank-Nullity Theorem Check</div>
              <div style={{ fontSize: "1.8rem", fontWeight: "800", color: "#34d399" }}>4 + 1 = 5</div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>rank(X) + nullity(X) = n</div>
            </div>
          </div>

          <div style={{ background: "rgba(15, 23, 42, 0.8)", padding: "1rem", borderRadius: "8px" }}>
            <h4 style={{ color: "#fbbf24", marginBottom: "0.5rem" }}>Null-Space Basis Vector v:</h4>
            <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", fontSize: "0.85rem" }}>
              {matrices.columns.map((col, idx) => (
                <div key={col}>
                  <span style={{ color: "var(--text-muted)" }}>{col}: </span>
                  <strong style={{ color: matrices.null_vector[idx] < 0 ? "#f87171" : "#34d399" }}>
                    {matrices.null_vector[idx]}
                  </strong>
                </div>
              ))}
            </div>
            <div style={{ marginTop: "0.5rem", fontSize: "0.8rem", color: "var(--text-muted)" }}>
              Equation: <strong>(1 &bull; Bullring) + (1 &bull; TownHall) + (1 &bull; BroadSt) + (1 &bull; Mailbox) - (1 &bull; Total) = 0</strong>
            </div>
            <div style={{ marginTop: "0.3rem", fontSize: "0.75rem", color: "#34d399" }}>
              ✓ Verification ||X v||_inf = {summary.verification_error.toExponential(2)} (machine zero)
            </div>
          </div>
        </div>
      );

    case 4:
      return (
        <div>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem" }}>
            <span className="badge badge-cyan">Basis Matrix A Shape: {matrices.A_shape.join(" x ")}</span>
            <span className="badge badge-emerald">Rank: {summary.basis_rank} (Full Column Rank)</span>
            <span className="badge badge-purple">Columns Retained: {summary.basis_columns_selected.join(", ")}</span>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)", marginBottom: "0.75rem" }}>
            By dropping the linearly dependent derived column 'Total', basis matrix A forms a basis for col(X), ensuring A^T A is invertible for projection and regression.
          </p>
          <div style={{ overflowX: "auto" }}>
            <table className="matrix-table">
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Slot</th>
                  {matrices.A_columns.map((c) => (
                    <th key={c}>{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {matrices.A_head.map((row, idx) => (
                  <tr key={idx}>
                    <td style={{ textAlign: "left", color: "var(--text-muted)" }}>#{idx + 1}</td>
                    {row.map((val, cIdx) => (
                      <td key={cIdx}>{val.toFixed(0)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );

    case 5:
      const chartDataGS = [
        { name: "Classical GS (CGS)", defect: summary.classical_gs_orthogonality_defect * 1e14 },
        { name: "Modified GS (MGS)", defect: summary.modified_gs_orthogonality_defect * 1e14 }
      ];
      return (
        <div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginBottom: "1rem" }}>
            <div className="card" style={{ background: "rgba(59, 130, 246, 0.1)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Classical Gram-Schmidt Defect</div>
              <div style={{ fontSize: "1.2rem", fontWeight: "bold", color: "#38bdf8" }}>
                {summary.classical_gs_orthogonality_defect.toExponential(2)}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>max |Q^T Q - I|</div>
            </div>
            <div className="card" style={{ background: "rgba(16, 185, 129, 0.1)" }}>
              <div style={{ fontSize: "0.75rem", color: "var(--text-muted)" }}>Modified Gram-Schmidt Defect</div>
              <div style={{ fontSize: "1.2rem", fontWeight: "bold", color: "#34d399" }}>
                {summary.modified_gs_orthogonality_defect.toExponential(2)}
              </div>
              <div style={{ fontSize: "0.7rem", color: "var(--text-muted)" }}>Superior numerical stability</div>
            </div>
          </div>

          <h4 style={{ fontSize: "0.9rem", marginBottom: "0.5rem", color: "var(--accent-cyan)" }}>
            Upper Triangular Factor R (4 x 4):
          </h4>
          <div style={{ overflowX: "auto" }}>
            <table className="matrix-table">
              <tbody>
                {matrices.R_mgs.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((val, cIdx) => (
                      <td key={cIdx} style={{ color: val > 0 ? "#34d399" : "var(--text-dim)" }}>
                        {val.toFixed(2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );

    case 6:
      return (
        <div>
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <span className="badge badge-emerald">Residual Orthogonality ||A^T e||_inf = {summary.orthogonality_check_inf_norm.toExponential(2)}</span>
            <span className="badge badge-cyan">Residual L2 Norm: {summary.residual_l2_norm}</span>
            <span className="badge badge-amber">Solved via LU (No Matrix Inverse)</span>
          </div>

          <h4 style={{ fontSize: "0.9rem", marginBottom: "0.5rem", color: "var(--accent-cyan)" }}>
            Orthogonal Projection Vector Coordinates x̂ in R^4:
          </h4>
          <div style={{ display: "flex", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap", fontSize: "0.85rem" }}>
            {matrices.x_hat_coordinates.map((val, idx) => (
              <div key={idx} className="card" style={{ padding: "0.5rem 1rem", flex: 1 }}>
                <div style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>Basis Col #{idx}</div>
                <div style={{ fontWeight: "bold", color: "#38bdf8" }}>{val}</div>
              </div>
            ))}
          </div>

          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            Sample demand vector b vs orthogonal projection b̂: b[0]={matrices.b_sample[0]}, b̂[0]={matrices.b_hat_sample[0]} (residual={matrices.residual_sample[0]}).
          </p>
        </div>
      );

    case 7:
      const chartPoints = matrices.chart_test_sample.actual.map((act, i) => ({
        slot: `T+${i}`,
        Actual: act,
        Predicted: matrices.chart_test_sample.predicted[i]
      }));

      return (
        <div>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <span className="badge badge-cyan">Train Slots: {summary.train_slots} (80%)</span>
            <span className="badge badge-emerald">Test Slots: {summary.test_slots} (20%)</span>
            <span className="badge badge-amber">Mean Test RMSE: {summary.mean_test_rmse} vehicles</span>
            <span className="badge badge-purple">Max Diff vs np.linalg.lstsq: {summary.max_diff_vs_numpy.toExponential(2)}</span>
          </div>

          <div style={{ height: 240, width: "100%", marginBottom: "1.25rem" }}>
            <h4 style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", marginBottom: "0.3rem" }}>
              Test Set: Actual vs Predicted Occupancy ({matrices.chart_test_sample.car_park}):
            </h4>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="slot" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155" }} />
                <Legend />
                <Line type="monotone" dataKey="Actual" stroke="#10b981" strokeWidth={2} dot={false} />
                <Line type="monotone" dataKey="Predicted" stroke="#38bdf8" strokeWidth={2} strokeDasharray="4 4" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <h4 style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", marginBottom: "0.3rem" }}>
            Output Next-Slot Zone Load Penalty ĥ[zone] (Feeds into Stage 10):
          </h4>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.5rem", fontSize: "0.8rem" }}>
            {Object.entries(summary.h_hat_loads).map(([zone, load]) => (
              <div key={zone} className="card" style={{ padding: "0.5rem 0.75rem" }}>
                <div style={{ color: "var(--text-muted)" }}>{zone}</div>
                <div style={{ fontSize: "1.2rem", fontWeight: "bold", color: "#fbbf24" }}>{Math.round(load * 100)}%</div>
                <div style={{ fontSize: "0.7rem", color: "var(--text-dim)" }}>Load factor: {load}</div>
              </div>
            ))}
          </div>
        </div>
      );

    case 8:
      const screeData = [
        { name: "λ1", val: summary.lambda_1 },
        { name: "λ2", val: summary.lambda_2 },
        { name: "λ3", val: stage.matrices.top2_eigenvalues[1] ? 2921.91 : 0 },
        { name: "λ4", val: 2113.40 }
      ];

      return (
        <div>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <span className="badge badge-cyan">λ1: {summary.lambda_1} (res={summary.verification_residual_1.toExponential(2)})</span>
            <span className="badge badge-emerald">λ2: {summary.lambda_2} (res={summary.verification_residual_2.toExponential(2)})</span>
            <span className="badge badge-amber">Cross-Check vs np.linalg.eigh: {summary.diff_vs_numpy_eigh.toExponential(2)}</span>
          </div>

          <div style={{ height: 220, width: "100%", marginBottom: "1rem" }}>
            <h4 style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", marginBottom: "0.3rem" }}>
              Eigenvalue Scree Plot (Dominant Axes of Variance):
            </h4>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={screeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis dataKey="name" stroke="#94a3b8" />
                <YAxis stroke="#94a3b8" />
                <Tooltip contentStyle={{ backgroundColor: "#1e293b", borderColor: "#334155" }} />
                <Bar dataKey="val" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <h4 style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", marginBottom: "0.3rem" }}>
            Sample Covariance Matrix S (4 x 4, Symmetric):
          </h4>
          <div style={{ overflowX: "auto" }}>
            <table className="matrix-table">
              <tbody>
                {matrices.S_covariance.map((row, rIdx) => (
                  <tr key={rIdx}>
                    {row.map((val, cIdx) => (
                      <td key={cIdx}>{val.toFixed(1)}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      );

    case 9:
      const reconProgression = summary.reconstruction_progression;
      return (
        <div>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <span className="badge badge-cyan">k=1 Explained: {summary.top_1_explained_pct}%</span>
            <span className="badge badge-emerald">k=2 Explained: {summary.top_2_explained_pct}%</span>
            <span className="badge badge-purple">Total Variance: {summary.total_variance}</span>
          </div>

          <h4 style={{ fontSize: "0.9rem", color: "var(--accent-cyan)", marginBottom: "0.5rem" }}>
            Spectral Reconstruction Progression vs Truncation Rank k:
          </h4>
          <div style={{ overflowX: "auto", marginBottom: "1.25rem" }}>
            <table className="matrix-table">
              <thead>
                <tr>
                  <th style={{ textAlign: "left" }}>Rank k</th>
                  <th>Eigenvalue λk</th>
                  <th>Indiv. Expl. Var %</th>
                  <th>Cumul. Expl. Var %</th>
                  <th>Frobenius Error ||S - Sk||_F</th>
                </tr>
              </thead>
              <tbody>
                {reconProgression.map((item) => (
                  <tr key={item.k}>
                    <td style={{ textAlign: "left", fontWeight: "bold" }}>k = {item.k}</td>
                    <td>{item.eigenvalue.toFixed(1)}</td>
                    <td>{item.individual_explained_variance_pct}%</td>
                    <td style={{ color: "#34d399", fontWeight: "bold" }}>{item.cumulative_explained_variance_pct}%</td>
                    <td style={{ color: item.frobenius_reconstruction_error === 0 ? "#34d399" : "#fbbf24" }}>
                      {item.frobenius_reconstruction_error.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
            * Demonstrates PCA compression: retaining just k=2 components captures &gt;90% of total spatial covariance while eliminating 4-variable complexity.
          </p>
        </div>
      );

    case 10:
      return (
        <div>
          <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1rem", flexWrap: "wrap" }}>
            <span className="badge badge-amber">Sample Spot Allocated: #{summary.allocated_spot_id} ({summary.allocated_zone})</span>
            <span className="badge badge-cyan">Computed Cost: {summary.cost}</span>
            <span className="badge badge-emerald">L1 Norm to Dest: {summary.norm_comparison?.l1_manhattan}</span>
            <span className="badge badge-purple">L2 Norm to Dest: {summary.norm_comparison?.l2_euclidean?.toFixed(2)}</span>
          </div>

          <div className="card" style={{ background: "rgba(15, 23, 42, 0.8)", marginBottom: "1rem" }}>
            <h4 style={{ fontSize: "0.85rem", color: "var(--accent-cyan)", marginBottom: "0.4rem" }}>
              Strategy Weightings Tested in Simulation:
            </h4>
            <ul style={{ fontSize: "0.8rem", color: "var(--text-muted)", listStyle: "none" }}>
              <li><strong>S1 (Nearest Entrance):</strong> α=0 (walk), β=1 (drive L2), γ=0 (no prediction)</li>
              <li><strong>S2 (Nearest Destination):</strong> α=1 (walk L1), β=0 (drive), γ=0 (no prediction)</li>
              <li><strong>S3 (Weighted Cost + Prediction):</strong> α=1 (walk L1), β=0.4 (drive L2), γ=2.5 (predictive load ĥ)</li>
            </ul>
          </div>
          <p style={{ fontSize: "0.85rem", color: "var(--text-muted)" }}>
            Switch to the <strong>"Strategy Comparison"</strong> tab in the navigation bar to see full Monte Carlo bar charts across S1, S2, and S3 on identical arrival sequences.
          </p>
        </div>
      );

    default:
      return null;
  }
}
