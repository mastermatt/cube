
import itertools


# Moves
Ux1 = 0
Ux2 = 1
Ux3 = 2
Rx1 = 3
Rx2 = 4
Rx3 = 5
Fx1 = 6
Fx2 = 7
Fx3 = 8
Dx1 = 9
Dx2 = 10
Dx3 = 11
Lx1 = 12
Lx2 = 13
Lx3 = 14
Bx1 = 15
Bx2 = 16
Bx3 = 17

# Facelets
U1 = 0
U2 = 1
U3 = 2
U4 = 3
U5 = 4
U6 = 5
U7 = 6
U8 = 7
U9 = 8
R1 = 9
R2 = 10
R3 = 11
R4 = 12
R5 = 13
R6 = 14
R7 = 15
R8 = 16
R9 = 17
F1 = 18
F2 = 19
F3 = 20
F4 = 21
F5 = 22
F6 = 23
F7 = 24
F8 = 25
F9 = 26
D1 = 27
D2 = 28
D3 = 29
D4 = 30
D5 = 31
D6 = 32
D7 = 33
D8 = 34
D9 = 35
L1 = 36
L2 = 37
L3 = 38
L4 = 39
L5 = 40
L6 = 41
L7 = 42
L8 = 43
L9 = 44
B1 = 45
B2 = 46
B3 = 47
B4 = 48
B5 = 49
B6 = 50
B7 = 51
B8 = 52
B9 = 53

# Colors
U = 0
R = 1
F = 2
D = 3
L = 4
B = 5

CORNER_FACELET = [
    [U9, R1, F3], [U7, F1, L3], [U1, L1, B3], [U3, B1, R3],
    [D3, F9, R7], [D1, L9, F7], [D7, B9, L7], [D9, R9, B7],
]

EDGE_FACELET = [
    [U6, R2], [U8, F2], [U4, L2], [U2, B2], [D6, R8], [D2, F8],
    [D4, L8], [D8, B8], [F6, R4], [F4, L6], [B6, L4], [B4, R6],
]

# C(n,k) is the binomial coefficient (n choose k)
CNK = [x[:] for x in [[0] * 12] * 12]  # int[12][12] needs to default to 0
FACT = [1] * 13  # fact[x] = x!
PERM_MULT = [x[:] for x in [[0] * 24] * 24]  # int[24][24]

FACES = ['U', 'R', 'F', 'D', 'L', 'B']

MOVE_2_STR = ["U", "U2", "U'", "R", "R2", "R'", "F", "F2", "F'",
              "D", "D2", "D'", "L", "L2", "L'", "B", "B2", "B'"]

STR_2_MOVE = {v: i for i, v in enumerate(MOVE_2_STR)}

UD_2_STD = [Ux1, Ux2, Ux3, Rx2, Fx2, Dx1, Dx2, Dx3, Lx2, Bx2]  # 0, 1, 2, 4, 7, 9, 10, 11, 13, 16
STD_2_UD = [0] * 18
# ^ only first 10 get hydrated, the rest seem to be to smooth index errors during lookup

CKMV2 = [x[:] for x in [[False] * 10] * 11]  # boolean[11][10]


def get_n_parity(idx: int, n: int) -> int:
    """
    static int getNParity(int idx, int n) {
        int p = 0;
        for (int i=n-2; i>=0; i--) {
            p ^= idx % (n-i);
            idx /= (n-i);
        }
        return p & 1;
    }
    """
    p = 0
    for i in range(2, n+1):
        p ^= idx % i
        idx //= i

    return p & 1


def get_8_perm(arr: list) -> int:
    """
    static int get8Perm(byte[] arr) {
        int idx = 0;
        int val = 0x76543210;
        for (int i=0; i<7; i++) {
            int v = arr[i] << 2;
            idx = (8 - i) * idx + ((val >> v) & 07);
            val -= 0x11111110 << v;
        }
        return idx;
    }
    """
    idx = 0
    val = 0x76543210
    for i in range(7):
        v = arr[i] << 2
        idx = (8 - i) * idx + ((val >> v) & 7)
        val -= 0x11111110 << v
    return idx


def set_8_perm(arr: list, idx: int):
    """
    Sets the first 8 indexes of the given list `arr` to a unique
    permutation of the digits [0-7] for the given int `idx`.

    ```
    arr1 = [None] * 8
    set_8_perm(arr1, idx)

    perms = [list(x) for x in itertools.permutations(range(8))]
    arr2 = perms[idx]

    arr1 == arr2  # True
    ```

    static void set8Perm(byte[] arr, int idx) {
        int val = 0x76543210;
        for (int i=0; i<7; i++) {
            int p = fact[7-i];
            int v = idx / p;
            idx -= v*p;
            v <<= 2;
            arr[i] = (byte) ((val >> v) & 07);
            int m = (1 << v) - 1;
            val = (val & m) + ((val >> 4) & ~m);
        }
        arr[7] = (byte)val;
    }

    0 <= idx < 40320 (8!)
    set_8_perm(arr, idx) == set_n_perm(arr, idx, 8)

    `set_8_perm(arr, idx)` has ~35% performance advantage over `set_n_perm(arr, idx, 8)`.
    """
    # assert 0 <= idx < 40320, idx
    val = 0x76543210
    for i in range(7):
        p = FACT[7 - i]
        v = idx // p
        idx -= p*v
        v <<= 2
        arr[i] = (val >> v) & 7
        mask = (1 << v) - 1
        val = (val & mask) + ((val >> 4) & ~mask)
    arr[7] = val


def get_n_perm(arr: list, n: int) -> int:
    """
    getNPerm(byte[] arr, int n) {
        int idx=0;
        for (int i=0; i<n; i++) {
            idx *= (n-i);
            for (int j=i+1; j<n; j++) {
                if (arr[j] < arr[i]) {
                    idx++;
                }
            }
        }
        return idx;
    }
    """
    idx = 0
    for i in range(n):
        idx *= n-i
        for j in range(i+1, n):
            if arr[j] < arr[i]:
                idx += 1
    return idx


def set_n_perm(arr: list, idx: int, n: int):
    """
    static void setNPerm(byte[] arr, int idx, int n) {
        arr[n-1] = 0;
        for (int i=n-2; i>=0; i--) {
            arr[i] = (byte) (idx % (n-i));
            idx /= (n-i);
            for (int j=i+1; j<n; j++) {
                if (arr[j] >= arr[i])
                    arr[j]++;
            }
        }
    }
    """
    # arr = [None] * n
    arr[-1] = 0
    for i in range(n-2, -1, -1):
        arr[i] = (idx % (n-i))
        idx //= (n - i)
        for j in range(i+1, n):
            if arr[j] >= arr[i]:
                arr[j] += 1
    # return arr


def get_comb(arr: list, mask: int) -> int:
    """
    static int getComb(byte[] arr, int mask) {
        int idxC = 0, idxP = 0, r = 4, val = 0x123;
        for (int i=11; i>=0; i--) {
            if ((arr[i] & 0xc) == mask) {
                int v = (arr[i] & 3) << 2;
                idxP = r * idxP + ((val >> v) & 0x0f);
                val -= 0x0111 >> (12-v);
                idxC += Cnk[i][r--];
            }
        }
        return idxP << 9 | (494 - idxC);
    }
    """
    idx_c = 0
    idx_p = 0
    r = 4
    val = 0x123
    for i in range(11, -1, -1):
        if (arr[i] & 0xc) == mask:
            v = (arr[i] & 3) << 2
            idx_p = r * idx_p + ((val >> v) & 0x0f)
            val -= 0x0111 >> (12 - v)
            idx_c += CNK[i][r]
            r -= 1
    return idx_p << 9 | (494 - idx_c)


def set_comb(arr: list, idx: int, mask: int):
    """
    static void setComb(byte[] arr, int idx, int mask) {
        int r = 4, fill = 11, val = 0x123;
        int idxC = 494 - (idx & 0x1ff);
        int idxP = idx >>> 9;
        for (int i=11; i>=0; i--) {
            if (idxC >= Cnk[i][r]) {
                idxC -= Cnk[i][r--];
                int p = fact[r & 3];
                int v = idxP / p << 2;
                idxP %= p;
                arr[i] = (byte) ((val >> v) & 3 | mask);
                int m = (1 << v) - 1;
                val = (val & m) + ((val >> 4) & ~m);
            } else {
                if ((fill & 0xc) == mask) {
                    fill -= 4;
                }
                arr[i] = (byte) (fill--);
            }
        }
    }

    js
    function setComb(arr, idx) {
        var fill, i, idxC, idxP, m, p, r, v, val;
        r = 4;
        fill = 11;
        val = 291;
        idxC = 494 - (idx & 511);
        idxP = idx >>> 9;
        for (i = 11; i >= 0; --i) {
            if (idxC >= Cnk[i][r]) {
                idxC -= Cnk[i][r--];
                p = fact[r & 3];
                v = ~~(idxP / p) << 2;
                idxP %= p;
                arr[i] = val >> v & 3 | 8;
                m = (1 << v) - 1;
                val = (val & m) + (val >> 4 & ~m);
            } else {
                (fill & 12) == 8 && (fill -= 4);
                arr[i] = fill--;
            }
        }
    }
    """
    r = 4
    fill = 11
    val = 0x123  # 291
    idx_c = 494 - (idx & 0x1ff)
    if idx < 0:
        raise ValueError(idx)
    idx_p = idx >> 9  # todo >>>
    for i in range(11, -1, -1):
        if idx_c >= CNK[i][r]:
            idx_c -= CNK[i][r]
            r -= 1
            p = FACT[r & 3]
            v = (idx_p // p) << 2
            idx_p %= p
            arr[i] = (val >> v) & 3 | mask
            m = (1 << v) - 1
            val = (val & m) + ((val >> 4) & ~m)
        else:
            if fill & 0xc == mask:
                fill -= 4
            arr[i] = fill
            fill -= 1


def to_cubie_cube(f: list, cc_ret):
    """
    static void toCubieCube(byte[] f, CubieCube ccRet) {
        byte ori;
        for (int i = 0; i < 8; i++)
            ccRet.cp[i] = 0;// invalidate corners
        for (int i = 0; i < 12; i++)
            ccRet.ep[i] = 0;// and edges
        byte col1, col2;
        for (byte i=0; i<8; i++) {
            // get the colors of the cubie at corner i, starting with U/D
            for (ori = 0; ori < 3; ori++)
                if (f[cornerFacelet[i][ori]] == U || f[cornerFacelet[i][ori]] == D)
                    break;
            col1 = f[cornerFacelet[i][(ori + 1) % 3]];
            col2 = f[cornerFacelet[i][(ori + 2) % 3]];

            for (byte j=0; j<8; j++) {
                if (col1 == cornerFacelet[j][1]/9 && col2 == cornerFacelet[j][2]/9) {
                    // in cornerposition i we have cornercubie j
                    ccRet.cp[i] = j;
                    ccRet.co[i] = (byte) (ori % 3);
                    break;
                }
            }
        }
        for (byte i=0; i<12; i++) {
            for (byte j=0; j<12; j++) {
                if (f[edgeFacelet[i][0]] == edgeFacelet[j][0]/9
                        && f[edgeFacelet[i][1]] == edgeFacelet[j][1]/9) {
                    ccRet.ep[i] = j;
                    ccRet.eo[i] = 0;
                    break;
                }
                if (f[edgeFacelet[i][0]] == edgeFacelet[j][1]/9
                        && f[edgeFacelet[i][1]] == edgeFacelet[j][0]/9) {
                    ccRet.ep[i] = j;
                    ccRet.eo[i] = 1;
                    break;
                }
            }
        }
    }
    """
    # invalidate corners and edges
    cc_ret.cp = [0]*8
    cc_ret.ep = [0]*12

    for i in range(8):
        # get the colors of the cubie at corner i, starting with U/D
        for ori in range(3):
            if f[CORNER_FACELET[i][ori]] in (U, D):
                break

        col1 = f[CORNER_FACELET[i][(ori + 1) % 3]]
        col2 = f[CORNER_FACELET[i][(ori + 2) % 3]]

        for j in range(8):
            if col1 == (CORNER_FACELET[j][1] // 9) and col2 == (CORNER_FACELET[j][2] // 9):
                # in corner position i we have corner cubie j
                cc_ret.cp[i] = j
                cc_ret.co[i] = ori % 3
                break

    for i in range(12):
        for j in range(12):
            if (f[EDGE_FACELET[i][0]] == EDGE_FACELET[j][0] // 9 and
                    f[EDGE_FACELET[i][1]] == EDGE_FACELET[j][1] // 9):
                cc_ret.ep[i] = j
                cc_ret.eo[i] = 0
                break

            if (f[EDGE_FACELET[i][0]] == EDGE_FACELET[j][1] // 9 and
                    f[EDGE_FACELET[i][1]] == EDGE_FACELET[j][0] // 9):
                cc_ret.ep[i] = j
                cc_ret.eo[i] = 1
                break


def to_face_cube(cc) -> str:
    """
    Returns the text notated version of the provided cubes current state.

    A solved cube would return "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"

    static String toFaceCube(CubieCube cc) {
        char[] f = new char[54];
        char[] ts = {'U', 'R', 'F', 'D', 'L', 'B'};
        for (int i=0; i<54; i++) {
            f[i] = ts[i/9];
        }
        for (byte c=0; c<8; c++) {
            byte j = cc.cp[c];// cornercubie with index j is at
            // cornerposition with index c
            byte ori = cc.co[c];// Orientation of this cubie
            for (byte n=0; n<3; n++)
                f[cornerFacelet[c][(n + ori) % 3]] = ts[cornerFacelet[j][n]/9];
        }
        for (byte e=0; e<12; e++) {
            byte j = cc.ep[e];// edgecubie with index j is at edgeposition
            // with index e
            byte ori = cc.eo[e];// Orientation of this cubie
            for (byte n=0; n<2; n++)
                f[edgeFacelet[e][(n + ori) % 2]] = ts[edgeFacelet[j][n]/9];
        }
        return new String(f);
    }
    :param cc:
    :return:
    """
    # start with a solved cube so the centers are populated
    result = [FACES[i//9] for i in range(54)]

    for i in range(8):
        j = cc.cp[i]  # corner cubie with index j is at corner position with index i
        ori = cc.co[i]  # orientation of this corner cubie
        for n in range(3):
            result[CORNER_FACELET[i][(n + ori) % 3]] = FACES[CORNER_FACELET[j][n] // 9]

    for i in range(12):
        j = cc.ep[i]  # edge cubie with index j is at edge position with index i
        ori = cc.eo[i]  # orientation of this edge cubie
        for n in range(2):
            result[EDGE_FACELET[i][(n + ori) % 2]] = FACES[EDGE_FACELET[j][n] // 9]

    return ''.join(result)


def _prep():
    for i in range(10):
        STD_2_UD[UD_2_STD[i]] = i

    for i in range(10):
        for j in range(10):
            ix = UD_2_STD[i]
            jx = UD_2_STD[j]
            CKMV2[i][j] = (ix // 3 == jx // 3) or ((ix // 3 % 3 == jx // 3 % 3) and (ix >= jx))

    for i in range(12):
        CNK[i][0] = CNK[i][i] = 1
        for j in range(1, i):
            CNK[i][j] = CNK[i - 1][j - 1] + CNK[i - 1][j]

    for i in range(12):
        FACT[i + 1] = FACT[i] * (i + 1)

    arr1 = [0]*4
    arr2 = [0]*4
    for i, j in itertools.product(range(24), range(24)):
        set_n_perm(arr1, i, 4)
        set_n_perm(arr2, j, 4)
        arr3 = [arr1[x] for x in arr2]
        PERM_MULT[i][j] = get_n_perm(arr3, 4)

_prep()
