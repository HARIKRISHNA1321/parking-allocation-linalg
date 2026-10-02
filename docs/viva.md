# University Viva Voce Defense Document: Parking Allocation Using Linear Algebra

**Project:** Parking Allocation Using Linear Algebra  
**Dataset:** UCI "Parking Birmingham" (id 482, CC BY 4.0, Stolfi 2017)  
**Implementation:** 100% from scratch in NumPy (no `np.linalg` routines used in production)

---

## 1. Stage-by-Stage Mathematical Summary (Real Run Numbers)

| Stage | Concept (Theory) | Purpose (Application) | Outcome (Real Run Values) |
|---|---|---|---|
| **1. Matrix Representation** | State representation in matrix spaces $\mathbb{R}^{T \times n}$ and $\mathbb{R}^{m \times p}$. | Structures temporal sensor data and spatial grid coordinates into computable mathematical arrays. | $X \in \mathbb{R}^{1307 \times 5}$ (4 car parks + Total); Lot $L \in \{0,1,2\}^{10 \times 12}$ (120 spots); Coordinates $P \in \mathbb{R}^{120 \times 2}$. |
| **2. RREF & LU Factorization** | Gaussian elimination with partial pivoting; $PA = LU$ factorization with forward/back substitution. | Identifies pivot columns and solves linear systems stably without explicit matrix inversion. | RREF pivot columns: $[0, 1, 2, 3]$; triangular solver residual $\|A x - b\|_\infty = 4.55 \times 10^{-13}$. |
| **3. Rank & Nullity** | Rank-Nullity Theorem: $\text{rank}(X) + \text{nullity}(X) = n$; $\text{null}(X) = \{v : X v = 0\}$. | Detects structural collinearity; non-zero null vectors identify redundant car park channels. | $\text{rank}(X) = 4, \text{nullity}(X) = 1$ ($4+1=5$); null vector $v = (1, 1, 1, 1, -1)^T$ with $\|X v\|_\infty = 0.00$; dependency is strictly **by construction** ($\text{Total} = \sum_{i=1}^4 c_i$). |
| **4. Basis Matrix** | Column space basis $\text{col}(A)$ as the minimal linearly independent spanning set of $\text{col}(X)$. | Drops redundant derived Total column to prevent singular normal equations in downstream projections. | Formed basis matrix $A \in \mathbb{R}^{1307 \times 4}$ using pivot columns $[0, 1, 2, 3]$; confirmed full column rank $\text{rank}(A) = 4$. |
| **5. Gram-Schmidt QR** | Orthonormalization: $A = QR$ where $Q^T Q = I$ and $R$ is upper triangular. | Decouples correlated features and stabilizes Euclidean geometric computations. | Classical GS defect $\max \|Q^T Q - I\| = 2.07 \times 10^{-14}$; Modified GS defect $= 2.60 \times 10^{-15}$ (MGS is $8\times$ more stable). |
| **6. Orthogonal Projection** | Projection $\hat{b} = A (A^T A)^{-1} A^T b$ onto $\text{col}(A)$ with residual $e = b - \hat{b} \perp \text{col}(A)$. | Filters unmodeled external demand noise using normal equations solved via custom LU. | Solved $(A^T A) \hat{x} = A^T b$ via LU solver; verified residual orthogonality $\|A^T (b - \hat{b})\|_\infty = 1.23 \times 10^{-7} \approx 0$; $\|e\|_2 = 1448.97$. |
| **7. Least Squares Forecasting** | Autoregressive OLS minimizing $\|Y - X \beta\|_2^2$ via normal equations $(X^T X)\beta = X^T Y$. | Forecasts next 30-min zone occupancies from lags $t-1, t-2$ to preemptively detect congestion. | Time-ordered 80/20 split (1044 train, 261 test); mean test RMSE $= 48.37$ vehicles; diff vs `np.linalg.lstsq` $= 7.39 \times 10^{-13}$; $\hat{h} = [0.292, 0.929, 0.720, 0.734]$. |
| **8. Eigendecomposition** | Spectral analysis: $S v = \lambda v$ on sample covariance matrix $S \in \mathbb{R}^{4 \times 4}$. | Discovers dominant principal directions of temporal parking demand across Birmingham. | Power iteration + Hotelling's deflation: $\lambda_1 = 35053.96$ ($\|S v_1 - \lambda_1 v_1\| < 10^{-11}$), $\lambda_2 = 11451.78$; matches `np.linalg.eigh` within $10^{-12}$. |
| **9. Diagonalization & PCA** | Spectral Theorem: $S = V \Lambda V^T$; rank-$k$ truncated reconstruction $S_k = \sum_{i=1}^k \lambda_i v_i v_i^T$. | Denoises and compresses multi-lot correlations into dominant principal components. | $\lambda_1$ explains $68.01\%$ variance; top 2 components explain $90.23\%$ variance; Frobenius error drops from $12006.13$ ($k=1$) to $3606.11$ ($k=2$) to $0.00$ ($k=4$). |
| **10. Allocation & Multi-Norms** | Multi-objective cost functional $C(v, s) = \alpha \|p_s - d\|_1 + \beta \|p_s - e\|_2 + \gamma \hat{h}[\text{zone}(s)]$. | Optimal vehicle-to-stall matching balancing walking ($L_1$), driving ($L_2$), and forecast congestion ($\hat{h}$). | Sample allocation: Spot #97 ($C = 5.475$); norm check: $L_1 = 1.0, L_2 = 1.00, L_\infty = 1.0$; S3 achieves $14.8\%$ zone std vs $32.1\%$ for S1 and $28.5\%$ for S2. |

---

## 2. Three-Minute Oral Viva / Live Demo Script

### **[Minute 0:00 - 0:45] Introduction & Real-World Dataset Grounding**
- **Examiner Action:** Open `http://127.0.0.1:8000` on Page 1 (Live Lot).
- **Spoken:**
  > *"Good morning, examiners. Our project solves urban parking allocation using pure linear algebra implemented completely from scratch in NumPy.*
  > *First, regarding our data grounding: we ingested the UCI Parking Birmingham dataset (id 482, 35,717 records across late 2016). By checking occupancy counts (mean 642, max 4327) against lot capacities (220 to 4675), we confirmed occupancy represents absolute vehicle counts, not fractions.*
  > *We extracted the 4 most complete car parks (Bullring, Town Hall, Broad Street, Mailbox) over 1307 half-hour intervals, handled the single missing entry via forward-fill for time continuity, and derived a 5th column Total. Because the dataset has no GPS stall coordinates, our 10x12 grid (120 spots) is our spatial model, partitioned into 4 zones whose stall counts—33, 22, 26, and 39—are exactly proportional to the real car park capacities."*

### **[Minute 0:45 - 1:45] The 10-Stage Linear Algebra Mathematical Pipeline**
- **Examiner Action:** Click to Page 2 (Math Pipeline). Navigate through Stages 2, 3, 5, 7, and 9.
- **Spoken:**
  > *"Moving to our mathematical pipeline, every single operation is implemented without black-box matrix solvers.*
  > *- In **Stage 2 & 3**, our from-scratch RREF with partial pivoting proves matrix X has Rank 4 and Nullity 1. The null-space basis vector is exactly (1, 1, 1, 1, -1), verifying that Total = Bullring + TownHall + BroadSt + Mailbox. This dependency is strictly by construction.*
  > *- In **Stage 4 & 5**, we drop the redundant column to get basis A. Running QR factorization, our Modified Gram-Schmidt achieves an orthogonality defect of 2.60e-15, outperforming Classical Gram-Schmidt (2.07e-14) due to immediate projection subtraction.*
  > *- In **Stage 6 & 7**, we perform orthogonal projection and autoregressive lag forecasting using our custom LU solver with forward/backward substitution—never inverting a matrix directly. Our regression parameters match np.linalg.lstsq down to 7.39e-13, producing next-slot zone load predictions h_hat.*
  > *- In **Stages 8 & 9**, power iteration with Hotelling's deflation computes the top covariance eigenvalues (35,053 and 11,451). Diagonalization proves just 2 principal components capture 90.23% of total citywide parking variance, compressing the system while cutting Frobenius reconstruction error by 70%."*

### **[Minute 1:45 - 2:30] Live Allocation & Distance Norms**
- **Examiner Action:** Return to Page 1. Select vehicle type `Electric Vehicle (EV)`, Destination `Town Hall`, Strategy `S3`. Click "Allocate Optimal Spot".
- **Spoken:**
  > *"Here on the Live Lot page, an incoming Electric Vehicle requests a stall near Town Hall. The system checks compatibility using the inner product with spot feature vectors: standard vehicles cannot take compact stalls, and EVs require active charging stalls. Incompatible stalls receive infinite cost.*
  > *For all valid spots, the system minimizes the linear functional:*
  > *C(v, s) = alpha * ||p - d||_1 + beta * ||p - e||_2 + gamma * h_hat[zone].*
  > *Notice the norm selection: we use the L1 Manhattan norm for pedestrian walking because drivers walk along perpendicular grid aisles, while the L2 Euclidean norm models vehicular entrance approach.*
  > *The system selects Spot #34, highlights it in gold, renders the top 3 alternative stalls, and displays the exact L1, L2, and Linf norm breakdown."*

### **[Minute 2:30 - 3:00] Strategy Benchmark & Conclusion**
- **Examiner Action:** Switch to Page 3 (Strategy Comparison). Click "Run Simulation".
- **Spoken:**
  > *"Finally, on Page 3, we run a Monte Carlo benchmark of 50 identical arrivals across three strategies:*
  > *- S1 (Nearest Entrance) minimizes driving distance but causes severe walking fatigue (mean 10.68 units).*
  > *- S2 (Nearest Destination) minimizes walking initially (4.12 units) but hyper-saturates destination quadrants, causing zone imbalance (std 28.5%).*
  > *- S3 (Weighted Cost with Stage 7 Prediction) balances walking distance (5.34 units) while achieving optimal zone distribution with a std dev of only 14.8%.*
  > *All 9 unit tests pass against NumPy and SymPy. The application runs as a unified FastAPI service serving the React build. Thank you, and I welcome your questions."*

---

## 3. Frequently Asked Examiner Viva Questions

### Q1: Why did you use Modified Gram-Schmidt instead of Classical Gram-Schmidt?
> **Answer:** Classical Gram-Schmidt computes all inner products $R_{ij} = q_i^T a_j$ against the original uncorrected column vector $a_j$. In finite-precision floating-point arithmetic, round-off errors accumulate, causing loss of orthogonality ($Q^T Q \neq I$). Modified Gram-Schmidt immediately projects each newly calculated orthogonal vector out of all remaining unnormalized vectors $v_j \leftarrow v_j - (q_i^T v_j) q_i$, achieving an orthogonality defect of $2.60 \times 10^{-15}$ compared to $2.07 \times 10^{-14}$ for CGS.

### Q2: Why solve normal equations using LU decomposition instead of matrix inversion?
> **Answer:** Computing the explicit matrix inverse $(A^T A)^{-1}$ is computationally inefficient ($O(n^3)$ with large constant factor) and numerically unstable due to floating-point roundoff and condition number squaring. LU decomposition with partial pivoting factors $A^T A = P^T L U$, reducing the solve to $O(n^2)$ triangular forward and backward substitution without matrix inversion.

### Q3: Why is the rank of matrix X exactly 4 and nullity 1?
> **Answer:** Matrix $X$ contains 5 columns: 4 real Birmingham car parks and 1 derived column 'Total'. Because $\text{Total} = \text{Col}_1 + \text{Col}_2 + \text{Col}_3 + \text{Col}_4$, column 5 is a linear combination of the first four columns. Since the 4 physical car parks are linearly independent, $\text{rank}(X) = 4$. By the Rank-Nullity Theorem, $\text{nullity}(X) = 5 - 4 = 1$, and the null vector is $(1, 1, 1, 1, -1)^T$. This dependency is strictly by construction.

### Q4: Why use the $L_1$ norm for walking distance and $L_2$ norm for driving distance?
> **Answer:** Parking lots and urban streets have grid structures with perpendicular aisles. Pedestrians must walk along right-angled paths, so their actual travel distance corresponds to the $L_1$ (Manhattan) metric $\|p - d\|_1 = |r_p - r_d| + |c_p - c_d|$. Driving from the entrance, however, allows diagonal transit across open lot driveways and perimeter corridors, modeled effectively by the Euclidean $L_2$ metric $\|p - e\|_2 = \sqrt{(r_p - r_e)^2 + (c_p - c_e)^2}$.
