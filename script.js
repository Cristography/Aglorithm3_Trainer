// --- Global DOM Ready ---
document.addEventListener('DOMContentLoaded', () => {
    // Strassen inputs
    createMatrixInputs(2, 'strassen_a', 'matrixA_strassen2x2'); createMatrixInputs(2, 'strassen_b', 'matrixB_strassen2x2');
    createMatrixInputs(4, 'strassen_a', 'matrixA_strassen4x4'); createMatrixInputs(4, 'strassen_b', 'matrixB_strassen4x4');

    // LUP inputs
    createMatrixInputs(3, 'lup_a', 'matrixA_lup3x3'); createVectorInputs(3, 'lup_b', 'vectorB_lup3x3');
    createMatrixInputs(4, 'lup_a', 'matrixA_lup4x4'); createVectorInputs(4, 'lup_b', 'vectorB_lup4x4');

    // FFT inputs
    createFFTInputs(4, 'fft_input_4');
    createFFTInputs(8, 'fft_input_8');

    // Initialize first tab
    const firstActiveTabButton = document.querySelector('.tab-button.active');
    if (firstActiveTabButton) {
        const activeTabId = firstActiveTabButton.getAttribute('onclick').match(/'([^']+)'/)[1];
        const activeTabContent = document.getElementById(activeTabId);
        if (activeTabContent) {
            activeTabContent.classList.add('active-content');
        } else {
            // Fallback if the active tab content isn't found, open the very first tab
            const firstTabButton = document.querySelector('.tab-button');
            if (firstTabButton) firstTabButton.click();
        }
    } else {
        // If no tab is marked active, activate the first one
        const firstTabButton = document.querySelector('.tab-button');
        if (firstTabButton) firstTabButton.click();
    }
});

// --- Tab Controls ---
function openTab(event, tabName) {
    const tabContents = document.getElementsByClassName('tab-content');
    for (let i = 0; i < tabContents.length; i++) {
        tabContents[i].classList.remove('active-content');
    }
    const tabButtons = document.getElementsByClassName('tab-button');
    for (let i = 0; i < tabButtons.length; i++) {
        tabButtons[i].classList.remove('active');
    }
    const currentTabContent = document.getElementById(tabName);
    if (currentTabContent) {
        currentTabContent.classList.add('active-content');
    }
    if (event && event.currentTarget) {
        event.currentTarget.classList.add('active');
    }
}

// --- Generic Input Creation ---
function createMatrixInputs(size, prefix, containerId) {
    const container = document.getElementById(containerId);
    if (!container) { console.error("Matrix Container not found:", containerId); return; }
    container.innerHTML = '';
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const input = document.createElement('input');
            input.type = 'number';
            input.id = `${prefix}_${size}_${i}${j}`;
            input.value = '0';
            input.title = `${prefix.split('_')[1]}[${i}][${j}]`;
            container.appendChild(input);
        }
    }
}

function createVectorInputs(size, prefix, containerId) {
    const container = document.getElementById(containerId);
    if (!container) { console.error("Vector Container not found:", containerId); return; }
    container.innerHTML = '';
    for (let i = 0; i < size; i++) {
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `${prefix}_${size}_${i}`;
        input.value = '0';
        input.title = `${prefix.split('_')[1]}[${i}]`;
        container.appendChild(input);
    }
}

function createFFTInputs(N, containerId) {
    const container = document.getElementById(containerId);
    if (!container) { console.error("FFT Container not found:", containerId); return; }
    container.innerHTML = '';
    for (let i = 0; i < N; i++) {
        const input = document.createElement('input');
        input.type = 'number';
        input.id = `fft_in_${N}_${i}`;
        input.value = '0';
        input.title = `x[${i}]`;
        container.appendChild(input);
    }
}

// --- Generic Value Getters ---
function getMatrixValues(prefix, size) {
    const matrix = [];
    for (let i = 0; i < size; i++) {
        matrix[i] = [];
        for (let j = 0; j < size; j++) {
            const inputElement = document.getElementById(`${prefix}_${size}_${i}${j}`);
            const val = inputElement ? inputElement.value : '0';
            matrix[i][j] = parseFloat(val) || 0;
        }
    }
    return matrix;
}

function getVectorValues(prefix, size) {
    const vector = [];
    for (let i = 0; i < size; i++) {
        const inputElement = document.getElementById(`${prefix}_${size}_${i}`);
        const val = inputElement ? inputElement.value : '0';
        vector[i] = parseFloat(val) || 0;
    }
    return vector;
}

function getFFTInputValues(N) {
    const samples = [];
    for (let i = 0; i < N; i++) {
        const inputElement = document.getElementById(`fft_in_${N}_${i}`);
        const realPart = parseFloat(inputElement ? inputElement.value : '0') || 0;
        samples.push({ re: realPart, im: 0 });
    }
    return samples;
}

// --- Random Value Fillers ---
function fillRandomStrassenValues(size) {
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const elA = document.getElementById(`strassen_a_${size}_${i}${j}`);
            if (elA) elA.value = Math.floor(Math.random() * 10);
            const elB = document.getElementById(`strassen_b_${size}_${i}${j}`);
            if (elB) elB.value = Math.floor(Math.random() * 10);
        }
    }
}

function fillRandomLUPValues(size) {
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const elA = document.getElementById(`lup_a_${size}_${i}${j}`);
            if (elA) elA.value = Math.floor(Math.random() * 11) - 5; // -5 to 5
        }
        const elB = document.getElementById(`lup_b_${size}_${i}`);
        if (elB) elB.value = Math.floor(Math.random() * 21) - 10; // -10 to 10
    }
}

function fillRandomFFTValues(N) {
    for (let i = 0; i < N; i++) {
        const el = document.getElementById(`fft_in_${N}_${i}`);
        if (el) el.value = Math.floor(Math.random() * 21) - 10; // -10 to 10
    }
}

// --- Strassen Algorithm Specific Helpers ---
function strassenAdd(A, B) {
    if (typeof A === 'number') return A + B;
    return [[A[0][0] + B[0][0], A[0][1] + B[0][1]],
    [A[1][0] + B[1][0], A[1][1] + B[1][1]]];
}
function strassenSubtract(A, B) {
    if (typeof A === 'number') return A - B;
    return [[A[0][0] - B[0][0], A[0][1] - B[0][1]],
    [A[1][0] - B[1][0], A[1][1] - B[1][1]]];
}
function strassenMultiply(A, B) {
    if (typeof A === 'number' && typeof B === 'number') return A * B;
    if (typeof A === 'number') {
        return [[A * B[0][0], A * B[0][1]], [A * B[1][0], A * B[1][1]]];
    }
    if (typeof B === 'number') {
        return [[B * A[0][0], B * A[0][1]], [B * A[1][0], B * A[1][1]]];
    }
    return [
        [A[0][0] * B[0][0] + A[0][1] * B[1][0], A[0][0] * B[0][1] + A[0][1] * B[1][1]],
        [A[1][0] * B[0][0] + A[1][1] * B[1][0], A[1][0] * B[0][1] + A[1][1] * B[1][1]]
    ];
}

// --- Strassen Main Calculation ---
function calculateStrassen(size) {
    const A_orig = getMatrixValues('strassen_a', size);
    const B_orig = getMatrixValues('strassen_b', size);

    let pValuesHTML = `<h4>P Values:</h4>`;
    let cSubMatricesHTML = ``;
    let finalCMatrix;

    const pValuesContainer = document.getElementById(`p_values_strassen${size}x${size}`);
    const cSubContainer = document.getElementById(`c_values_strassen${size}x${size}_sub`);
    const finalCContainer = document.getElementById(`matrixC_strassen${size}x${size}_output`);

    if (pValuesContainer) pValuesContainer.innerHTML = '';
    if (cSubContainer) cSubContainer.innerHTML = '';
    if (finalCContainer) finalCContainer.innerHTML = '';


    if (size === 2) {
        const a11 = A_orig[0][0], a12 = A_orig[0][1], a21 = A_orig[1][0], a22 = A_orig[1][1];
        const b11 = B_orig[0][0], b12 = B_orig[0][1], b21 = B_orig[1][0], b22 = B_orig[1][1];

        const p1 = strassenMultiply(strassenAdd(a11, a22), strassenAdd(b11, b22));
        const p2 = strassenMultiply(strassenAdd(a21, a22), b11);
        const p3 = strassenMultiply(a11, strassenSubtract(b12, b22));
        const p4 = strassenMultiply(a22, strassenSubtract(b21, b11));
        const p5 = strassenMultiply(strassenAdd(a11, a12), b22);
        const p6 = strassenMultiply(strassenSubtract(a21, a11), strassenAdd(b11, b12));
        const p7 = strassenMultiply(strassenSubtract(a12, a22), strassenAdd(b21, b22));

        pValuesHTML += `<p><span>P1</span> = (a11+a22)(b11+b22) = (${a11}+${a22})(${b11}+${b22}) = ${p1.toFixed(4)}</p>`;
        pValuesHTML += `<p><span>P2</span> = (a21+a22)b11 = (${a21}+${a22})(${b11}) = ${p2.toFixed(4)}</p>`;
        pValuesHTML += `<p><span>P3</span> = a11(b12-b22) = (${a11})(${b12}-${b22}) = ${p3.toFixed(4)}</p>`;
        pValuesHTML += `<p><span>P4</span> = a22(b21-b11) = (${a22})(${b21}-${b11}) = ${p4.toFixed(4)}</p>`;
        pValuesHTML += `<p><span>P5</span> = (a11+a12)b22 = (${a11}+${a12})(${b22}) = ${p5.toFixed(4)}</p>`;
        pValuesHTML += `<p><span>P6</span> = (a21-a11)(b11+b12) = (${a21}-${a11})(${b11}+${b12}) = ${p6.toFixed(4)}</p>`;
        pValuesHTML += `<p><span>P7</span> = (a12-a22)(b21+b22) = (${a12}-${a22})(${b21}+${b22}) = ${p7.toFixed(4)}</p>`;


        const c11 = strassenAdd(strassenSubtract(strassenAdd(p1, p4), p5), p7);
        const c12 = strassenAdd(p3, p5);
        const c21 = strassenAdd(p2, p4);
        const c22 = strassenAdd(strassenSubtract(strassenAdd(p1, p3), p2), p6);

        finalCMatrix = [[c11, c12], [c21, c22]];
        if (pValuesContainer) pValuesContainer.innerHTML = pValuesHTML;
        displayMatrixOutputGeneral(finalCMatrix, finalCContainer, size, "Matrix C (Result)");

    } else if (size === 4) {
        const A11 = [[A_orig[0][0], A_orig[0][1]], [A_orig[1][0], A_orig[1][1]]];
        const A12 = [[A_orig[0][2], A_orig[0][3]], [A_orig[1][2], A_orig[1][3]]];
        const A21 = [[A_orig[2][0], A_orig[2][1]], [A_orig[3][0], A_orig[3][1]]];
        const A22 = [[A_orig[2][2], A_orig[2][3]], [A_orig[3][2], A_orig[3][3]]];

        const B11 = [[B_orig[0][0], B_orig[0][1]], [B_orig[1][0], B_orig[1][1]]];
        const B12 = [[B_orig[0][2], B_orig[0][3]], [B_orig[1][2], B_orig[1][3]]];
        const B21 = [[B_orig[2][0], B_orig[2][1]], [B_orig[3][0], B_orig[3][1]]];
        const B22 = [[B_orig[2][2], B_orig[2][3]], [B_orig[3][2], B_orig[3][3]]];

        const P1 = strassenMultiply(strassenAdd(A11, A22), strassenAdd(B11, B22));
        const P2 = strassenMultiply(strassenAdd(A21, A22), B11);
        const P3 = strassenMultiply(A11, strassenSubtract(B12, B22));
        const P4 = strassenMultiply(A22, strassenSubtract(B21, B11));
        const P5 = strassenMultiply(strassenAdd(A11, A12), B22);
        const P6 = strassenMultiply(strassenSubtract(A21, A11), strassenAdd(B11, B12));
        const P7 = strassenMultiply(strassenSubtract(A12, A22), strassenAdd(B21, B22));

        const P_Matrices = { P1, P2, P3, P4, P5, P6, P7 };
        for (const pName in P_Matrices) {
            pValuesHTML += `<p><span>${pName}</span> = ${formatStrassenSubMatrixForDisplay(P_Matrices[pName])}</p>`;
        }

        const C11 = strassenAdd(strassenSubtract(strassenAdd(P1, P4), P5), P7);
        const C12 = strassenAdd(P3, P5);
        const C21 = strassenAdd(P2, P4);
        const C22 = strassenAdd(strassenSubtract(strassenAdd(P1, P3), P2), P6);

        cSubMatricesHTML = `<h4>C Sub-Matrices:</h4>`;
        cSubMatricesHTML += `<p><span>C11</span> = ${formatStrassenSubMatrixForDisplay(C11)}</p>`;
        cSubMatricesHTML += `<p><span>C12</span> = ${formatStrassenSubMatrixForDisplay(C12)}</p>`;
        cSubMatricesHTML += `<p><span>C21</span> = ${formatStrassenSubMatrixForDisplay(C21)}</p>`;
        cSubMatricesHTML += `<p><span>C22</span> = ${formatStrassenSubMatrixForDisplay(C22)}</p>`;

        if (pValuesContainer) pValuesContainer.innerHTML = pValuesHTML;
        if (cSubContainer) cSubContainer.innerHTML = cSubMatricesHTML;

        finalCMatrix = [
            [C11[0][0], C11[0][1], C12[0][0], C12[0][1]],
            [C11[1][0], C11[1][1], C12[1][0], C12[1][1]],
            [C21[0][0], C21[0][1], C22[0][0], C22[0][1]],
            [C21[1][0], C21[1][1], C22[1][0], C22[1][1]]
        ];
        displayMatrixOutputGeneral(finalCMatrix, finalCContainer, size, "Matrix C (Result)");
    }
}
function formatStrassenSubMatrixForDisplay(matrix) {
    if (!matrix || !matrix[0] || !matrix[1] || matrix[0].length < 2 || matrix[1].length < 2) {
        return `<div class="sub-matrix-display">Error: Invalid sub-matrix data</div>`;
    }
    return `<div class="sub-matrix-display">
                <table>
                    <tr><td>${matrix[0][0].toFixed(4)}</td><td>${matrix[0][1].toFixed(4)}</td></tr>
                    <tr><td>${matrix[1][0].toFixed(4)}</td><td>${matrix[1][1].toFixed(4)}</td></tr>
                </table>
            </div>`;
}

// --- LUP Algorithm Specific Helpers ---
function createZeroMatrix(size) { return Array.from({ length: size }, () => Array(size).fill(0)); }
function createIdentityMatrix(size) {
    const I = createZeroMatrix(size);
    for (let i = 0; i < size; i++) I[i][i] = 1;
    return I;
}

function lupDecomposition(matrixA_orig, size) {
    let A = matrixA_orig.map(row => [...row]);
    let P_vec = Array.from({ length: size }, (_, i) => i);
    let L = createIdentityMatrix(size);
    let U = createZeroMatrix(size);
    let swapCount = 0;
    const tol = 1e-9; // Tolerance for pivot check

    for (let k = 0; k < size; k++) {
        let maxVal = 0;
        let pivotRow = k;
        for (let i = k; i < size; i++) {
            if (Math.abs(A[i][k]) > maxVal) {
                maxVal = Math.abs(A[i][k]);
                pivotRow = i;
            }
        }
        if (maxVal < tol) return { error: "Singular matrix (zero or too small pivot encountered)." };

        if (pivotRow !== k) {
            [A[k], A[pivotRow]] = [A[pivotRow], A[k]];
            [P_vec[k], P_vec[pivotRow]] = [P_vec[pivotRow], P_vec[k]];
            // Swap corresponding rows in L for elements already computed (j < k)
            for (let col = 0; col < k; col++) {
                [L[k][col], L[pivotRow][col]] = [L[pivotRow][col], L[k][col]];
            }
            swapCount++;
        }

        U[k][k] = A[k][k]; // Pivot element
        for (let j = k + 1; j < size; j++) U[k][j] = A[k][j]; // Row k of U

        if (Math.abs(U[k][k]) < tol) return { error: "Division by zero (or too small pivot) for L multiplier." };
        for (let i = k + 1; i < size; i++) {
            L[i][k] = A[i][k] / U[k][k];
            for (let j = k + 1; j < size; j++) A[i][j] -= L[i][k] * U[k][j];
        }
    }
    let P_matrix = createZeroMatrix(size);
    for (let i = 0; i < size; i++) P_matrix[i][P_vec[i]] = 1; // P_matrix[row_idx_after_permutation][original_row_idx] = 1
    // To permute rows of A to get PA: for each i, row i of PA is row P_vec[i] of A_original.
    // Or, for the P matrix such that PA=LU: P[i][j] = 1 if row j of A moves to row i of PA.
    // So, P[i][P_vec[i]] = 1 is correct.
    // However, the P from PA=LU is such that P acting on A gives permuted A.
    // The P_vec stores where original row i went.
    // P_matrix should be constructed such that P_matrix * A = A_permuted.
    // P_matrix[k] should be the P_vec[k]-th unit vector. No, this isn't right.
    // If P_vec[k] = m, it means original row m moved to row k.
    // So, row k of P_matrix should have a 1 at column m.  P_matrix[k][m] = 1.
    // P_matrix[k][P_vec[k]] = 1. This means P_vec stores column index for P_matrix. Correct.
    P_matrix = createZeroMatrix(size);
    for (let i = 0; i < size; i++) {
        P_matrix[i][P_vec[i]] = 1;
    }


    return { L, U, P_matrix, P_vector: P_vec, error: null, swapCount };
}
function permuteVector(vectorB_orig, P_vector) {
    const size = vectorB_orig.length;
    let PB = new Array(size).fill(0);
    // P_vector[i] = j means original row j moved to position i in the permuted matrix/vector.
    // So, PB[i] should be B_original[P_vector[i]].
    for (let i = 0; i < size; i++) PB[i] = vectorB_orig[P_vector[i]];
    return PB;
}
function forwardSubstitution(L, Pb) {
    const size = Pb.length;
    let y = new Array(size).fill(0);
    const tol = 1e-9;
    for (let i = 0; i < size; i++) {
        let sum = 0;
        for (let j = 0; j < i; j++) sum += L[i][j] * y[j];
        if (Math.abs(L[i][i]) < tol) return { error: "Division by zero in L (forward sub)." };
        y[i] = (Pb[i] - sum) / L[i][i];
    }
    return y;
}
function backwardSubstitution(U, y) {
    const size = y.length;
    let x = new Array(size).fill(0);
    const tol = 1e-9;
    for (let i = size - 1; i >= 0; i--) {
        let sum = 0;
        for (let j = i + 1; j < size; j++) sum += U[i][j] * x[j];
        if (Math.abs(U[i][i]) < tol) return { error: "Division by zero in U (backward sub)." };
        x[i] = (y[i] - sum) / U[i][i];
    }
    return x.map(val => parseFloat(val.toFixed(4))); // Keep fixed precision for final X
}

// --- LUP Main Calculation ---
function calculateLUP(size) {
    const matrixA = getMatrixValues(`lup_a`, size);
    const vectorB = getVectorValues(`lup_b`, size);
    const errorDiv = document.getElementById(`error_lup${size}x${size}`);
    if (errorDiv) errorDiv.textContent = '';

    ['P_matrix', 'L_matrix', 'U_matrix', 'PB_vector', 'Y_vector', 'X_vector'].forEach(idPart => {
        const el = document.getElementById(`${idPart}_lup${size}x${size}`);
        if (el) el.innerHTML = '';
    });

    const decomp = lupDecomposition(matrixA, size);
    if (decomp.error) {
        if (errorDiv) errorDiv.textContent = decomp.error;
        return;
    }
    const { L, U, P_matrix, P_vector } = decomp;

    const PB = permuteVector(vectorB, P_vector);
    const Y_solution = forwardSubstitution(L, PB);
    if (Y_solution.error) {
        if (errorDiv) errorDiv.textContent = Y_solution.error;
        return;
    }
    const X_solution = backwardSubstitution(U, Y_solution);
    if (X_solution.error) {
        if (errorDiv) errorDiv.textContent = X_solution.error;
        return;
    }

    // Construct P_matrix for PA=LU from P_vector (where P_vector[i] is the original row index that moves to row i)
    let P_display_matrix = createZeroMatrix(size);
    for (let i = 0; i < size; i++) {
        P_display_matrix[i][P_vector[i]] = 1;
    }

    displayMatrixOutputGeneral(P_display_matrix, document.getElementById(`P_matrix_lup${size}x${size}`), size, `Permutation Matrix P (PA = LU)`);
    displayMatrixOutputGeneral(L, document.getElementById(`L_matrix_lup${size}x${size}`), size, `Lower Triangular Matrix L`);
    displayMatrixOutputGeneral(U, document.getElementById(`U_matrix_lup${size}x${size}`), size, `Upper Triangular Matrix U`);
    displayVectorOutputGeneral(PB, document.getElementById(`PB_vector_lup${size}x${size}`), `Permuted Vector PB (from B, s.t. LY = PB)`);
    displayVectorOutputGeneral(Y_solution, document.getElementById(`Y_vector_lup${size}x${size}`), `Intermediate Vector Y (s.t. UX = Y)`);
    displayVectorOutputGeneral(X_solution, document.getElementById(`X_vector_lup${size}x${size}`), `Solution Vector X (Ax = B)`);
}


// --- FFT Algorithm Specific Helpers ---
const Complex = {
    add: (a, b) => ({ re: a.re + b.re, im: a.im + b.im }),
    sub: (a, b) => ({ re: a.re - b.re, im: a.im - b.im }),
    mul: (a, b) => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }),
    exp: (phi) => ({ re: Math.cos(phi), im: Math.sin(phi) }),
    toString: (c, dp = 4) => {
        const reStr = parseFloat(c.re.toFixed(dp));
        const imStrAbs = parseFloat(Math.abs(c.im).toFixed(dp));
        const imSign = c.im < -1e-9 ? '-' : '+'; // Check against small epsilon for -0.0000 cases

        if (Math.abs(reStr) < 1e-9 && Math.abs(imStrAbs) < 1e-9) return `0`; // Both effectively zero
        if (Math.abs(imStrAbs) < 1e-9) return `${reStr}`; // Purely real
        if (Math.abs(reStr) < 1e-9) return `${c.im < 0 ? '-' : ''}j ${imStrAbs}`; // Purely imaginary

        return `${reStr} ${imSign} j ${imStrAbs}`;
    }
};
let fftStepLog = [];

function fft_dit_recursive(x) {
    const N = x.length;
    fftStepLog.push(`FFT N=${N}, In: [${x.map(c => Complex.toString(c, 2)).join(', ')}]`);
    if (N === 1) {
        fftStepLog.push(`  Base N=1, Out: ${Complex.toString(x[0], 2)}`);
        return [x[0]];
    }
    if (N % 2 !== 0) throw new Error("FFT size must be a power of 2.");

    const even = [], odd = [];
    for (let i = 0; i < N / 2; i++) { even.push(x[2 * i]); odd.push(x[2 * i + 1]); }

    const X_even = fft_dit_recursive(even);
    const X_odd = fft_dit_recursive(odd);

    fftStepLog.push(`  Res N=${N}: E:[${X_even.map(c => Complex.toString(c, 2)).join(', ')}] O:[${X_odd.map(c => Complex.toString(c, 2)).join(', ')}]`);

    const X = new Array(N);
    for (let k = 0; k < N / 2; k++) {
        const phi = -2 * Math.PI * k / N;
        const twiddle = Complex.exp(phi);
        const term = Complex.mul(twiddle, X_odd[k]);
        X[k] = Complex.add(X_even[k], term);
        X[k + N / 2] = Complex.sub(X_even[k], term);
        fftStepLog.push(`    k=${k}: W=${Complex.toString(twiddle, 2)}, T=${Complex.toString(term, 2)} => X[${k}]=${Complex.toString(X[k], 2)}, X[${k + N / 2}]=${Complex.toString(X[k + N / 2], 2)}`);
    }
    return X;
}

// --- FFT Main Calculation ---
function calculateFFT(N) {
    const inputSamples = getFFTInputValues(N);
    const outputDisplay = document.getElementById(`fft_output_${N}`);
    const stepsDisplay = document.getElementById(`fft_steps_${N}`);

    if (outputDisplay) outputDisplay.innerHTML = '';
    if (stepsDisplay) stepsDisplay.innerHTML = '';
    fftStepLog = [];

    try {
        const Xk = fft_dit_recursive(inputSamples);

        let resultHTML = `<h5 class="fft-result-title">FFT Output X[k]:</h5>
                          <table class="fft-results-table">
                            <thead>
                                <tr>
                                    <th>k</th>
                                    <th>X[k] (Complex)</th>
                                    <th>Real Part</th>
                                    <th>Imaginary Part</th>
                                </tr>
                            </thead>
                            <tbody>`;
        Xk.forEach((val, k) => {
            const magnitude = Math.sqrt(val.re * val.re + val.im * val.im);
            const phase = Math.atan2(val.im, val.re);
            resultHTML += `
                <tr>
                    <td>${k}</td>
                    <td>${Complex.toString(val, 4)}</td>
                    <td>${val.re.toFixed(4)}</td>
                    <td>${val.im.toFixed(4)}</td>
                </tr>
            `;
        });
        resultHTML += `</tbody></table>`;
        if (outputDisplay) outputDisplay.innerHTML = resultHTML;

        if (N <= 8) {
            if (stepsDisplay) stepsDisplay.innerHTML = `<h5>Calculation Steps (DIT Algorithm):</h5><pre class="fft-steps-pre">${fftStepLog.join("\n")}</pre>`;
        } else {
            if (stepsDisplay) stepsDisplay.innerHTML = `<p class="fft-steps-notice"><i>Detailed calculation steps are shown for N ≤ 8.</i></p>`;
        }
    } catch (error) {
        if (outputDisplay) outputDisplay.innerHTML = `<p class="error-message">Error computing FFT: ${error.message}</p>`;
        console.error("FFT Error:", error);
    }
}


// --- Generic Display Functions ---
function displayMatrixOutputGeneral(matrix, container, size, title = null) {
    if (!container) return;
    let html = title ? `<h4>${title}:</h4>` : '';
    let gridClass = `grid-${size}x${size}`;

    html += `<div class="matrix-grid ${gridClass} output-grid">`;
    for (let i = 0; i < size; i++) {
        for (let j = 0; j < size; j++) {
            const val = (matrix && matrix[i] && typeof matrix[i][j] === 'number') ? parseFloat(matrix[i][j].toFixed(4)) : 'N/A';
            html += `<div class="cell-value">${val}</div>`;
        }
    }
    html += `</div>`;
    container.innerHTML = html;
}

function displayVectorOutputGeneral(vector, container, title = null) {
    if (!container) return;
    let html = title ? `<h4>${title}:</h4>` : '';
    const size = vector ? vector.length : 0;
    if (size === 0) {
        container.innerHTML = html + `<p>Vector data not available.</p>`;
        return;
    }
    let gridClass = `grid-${size}x1`;

    html += `<div class="vector-grid ${gridClass} output-grid">`;
    for (let i = 0; i < size; i++) {
        const val = (typeof vector[i] === 'number') ? parseFloat(vector[i].toFixed(4)) : 'N/A';
        html += `<div class="cell-value">${val}</div>`;
    }
    html += `</div>`;
    container.innerHTML = html;
}