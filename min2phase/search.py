"""
The Two-Phase Algorithm

http://kociemba.org/math/twophase.htm
http://kociemba.org/math/imptwophase.htm
"""
import collections
import time

from . import coord_cube
from . import cubie_cube
from . import tools
from . import util


class Search(object):
    """
    Rubik's Cube Solver.

    A much faster and smaller implementation of Two-Phase Algorithm.
    Symmetry is used to reduce memory used.

    @author Shuang Chen (original Java implementation)
    """

    def __init__(self):
        self.move_set_count = 0
        self.move = [0] * 31
        self.phase1_move = [None] * 12
        self.phase2_move = [None] * 18

        self.corn = [0] * 20
        self.mid4 = [0] * 20
        self.ud8e = [0] * 20

        self.twist = [0] * 6
        self.flip = [0] * 6
        self.slice = [0] * 6

        self.corn0 = [0] * 6
        self.ud8e0 = [0] * 6
        self.prune = [0] * 6

        self.urf_idx: int = None
        self.depth1: int = None
        self.max_depth2: int = None
        self.sol: int = None
        self.valid1: int = 0
        self.valid2: int = 0
        self.solution: str = None
        self.time_out: int = None
        self.time_min: int = None

        self.append_length = False
        self.inverse_solution = False
        self.use_separator = False

        self.first_axis_restriction: int = None
        self.last_axis_restriction: int = None
        self.cc = cubie_cube.CubieCube()

    def get_solution(
        self,
        facelets,
        max_depth,
        time_out,
        time_min,
        append_length=False,
        inverse_solution=False,
        use_separator=False,
        first_axis_restriction_str=None,
        last_axis_restriction_str=None,
    ):
        """
        Computes the solver string for a given cube.

        @param facelets
                is the cube definition string format.
                The names of the facelet positions of the cube:
         <pre>
                     |************|
                     |*U1**U2**U3*|
                     |************|
                     |*U4**U5**U6*|
                     |************|
                     |*U7**U8**U9*|
                     |************|
         ************|************|************|************|
         *L1**L2**L3*|*F1**F2**F3*|*R1**R2**F3*|*B1**B2**B3*|
         ************|************|************|************|
         *L4**L5**L6*|*F4**F5**F6*|*R4**R5**R6*|*B4**B5**B6*|
         ************|************|************|************|
         *L7**L8**L9*|*F7**F8**F9*|*R7**R8**R9*|*B7**B8**B9*|
         ************|************|************|************|
                     |************|
                     |*D1**D2**D3*|
                     |************|
                     |*D4**D5**D6*|
                     |************|
                     |*D7**D8**D9*|
                     |************|
         </pre>

         A cube definition string "UBL..." means for example: In position U1 we have the
         U-color, in position U2 we have the B-color, in position U3 we have the L color etc.
         according to the order U1, U2, U3, U4, U5, U6, U7, U8, U9, R1, R2, R3, R4, R5, R6, R7,
         R8, R9, F1, F2, F3, F4, F5, F6, F7, F8, F9, D1, D2, D3, D4, D5, D6, D7, D8, D9, L1, L2,
         L3, L4, L5, L6, L7, L8, L9, B1, B2, B3, B4, B5, B6, B7, B8, B9 of the enum constants.


        @param max_depth
                defines the maximal allowed maneuver length. For random cubes, a maxDepth of 21
                usually will return a solution in less than 0.02 seconds on average. With a
                maxDepth of 20 it takes about 0.1 seconds on average to find a solution,
                but it may take much longer for specific cubes.

        @param time_out
                defines the maximum computing time of the method in milliseconds. If it does not
                return with a solution, it returns with an error code.

        @param time_min
                defines the minimum computing time of the method in milliseconds. So,
                if a solution is found within given time, the computing will continue to find
                shorter solution(s). Btw, if timeMin > timeOut, timeMin will be set to timeOut.

        @param append_length
                determines if a tag such as "(21f)" will be appended to the solution.

        @param inverse_solution
                determines if the solution will be inversed to a scramble/state generator.

        @param use_separator
                determines if a " . " separates the phase1 and phase2 parts of
                the solver string like in F' R B R L2 F . U2 U D for example.

        @param first_axis_restriction_str
                  The solution generated will not start by turning
                  any face on the axis of firstAxisRestrictionStr.

        @param last_axis_restriction_str
                  The solution generated will not end by turning
                  any face on the axis of lastAxisRestrictionStr.

        @return The solution string

        @raises ValueError
                Error 1: There is not exactly one facelet of each colour
                Error 2: Not all 12 edges exist exactly once
                Error 3: Flip error: One edge has to be flipped
                Error 4: Not all corners exist exactly once
                Error 5: Twist error: One corner has to be twisted
                Error 6: Parity error: Two corners or two edges have to be exchanged
                Error 7: No solution exists for the given maxDepth
                Error 8: Timeout, no solution within given time
                Error 9: Invalid firstAxisRestrictionStr or lastAxisRestrictionStr

        """
        self.sol = max_depth + 1
        self.time_out = time.time() + time_out
        self.time_min = self.time_out + min(time_min - time_out, 0)

        self.append_length = append_length
        self.inverse_solution = inverse_solution
        self.use_separator = use_separator

        self.first_axis_restriction = None
        self.last_axis_restriction = None

        self.solution = None

        if first_axis_restriction_str is not None:
            if first_axis_restriction_str not in util.STR_2_MOVE:
                raise ValueError("Error 9: invalid first axis restriction value")

            self.first_axis_restriction = util.STR_2_MOVE[first_axis_restriction_str]

            if self.first_axis_restriction % 3 != 0:
                raise ValueError("Error 9: first axis restriction not divisible by 3")

            if self.first_axis_restriction < 9:
                # firstAxisRestriction defines an axis of turns that
                # aren't permitted. Make sure we restrict the entire
                # axis, and not just one of the faces. See the axis
                # filtering in phase1() for more details.
                self.first_axis_restriction += 9

        if last_axis_restriction_str is not None:
            if last_axis_restriction_str not in util.STR_2_MOVE:
                raise ValueError("Error 9: invalid last axis restriction value")

            self.last_axis_restriction = util.STR_2_MOVE[last_axis_restriction_str]

            if self.last_axis_restriction % 3 != 0:
                raise ValueError("Error 9: last axis restriction not divisible by 3")

            if self.last_axis_restriction < 9:
                # lastAxisRestriction defines an axis of turns that
                # aren't permitted. Make sure we restrict the entire
                # axis, and not just one of the faces. See the axis
                # filtering in phase1() for more details.
                self.last_axis_restriction += 9

        self.verify(facelets)
        tools.init_index()
        s = time.monotonic()
        solution = self._solve()
        print(time.monotonic() - s)
        return solution

    def verify(self, facelets: str):
        """
        int verify(String facelets) {
            int count = 0x000000;
            try {
                String center = new String(new char[] {
                    facelets.charAt(4),
                    facelets.charAt(13),
                    facelets.charAt(22),
                    facelets.charAt(31),
                    facelets.charAt(40),
                    facelets.charAt(49)
                });
                for (int i=0; i<54; i++) {
                    f[i] = (byte) center.indexOf(facelets.charAt(i));
                    if (f[i] == -1) {
                        return -1;
                    }
                    count += 1 << (f[i] << 2);
                }
            } catch (Exception e) {
                return -1;
            }
            if (count != 0x999999) {
                return -1;
            }
            Util.toCubieCube(f, cc);
            return cc.verify();
        }
        """
        if len(facelets) != 54:
            raise ValueError("Invalid facelet length", facelets)

        centers = [
            facelets[4],
            facelets[13],
            facelets[22],
            facelets[31],
            facelets[40],
            facelets[49],
        ]

        counter = collections.Counter(facelets)

        if len(counter) != 6:
            raise ValueError("There are not exactly 6 unique notations", facelets, counter)

        if counter.keys() != set(centers):
            raise ValueError(
                "The facelets and the centers do not share a common, distinct set of notations.",
                facelets,
                centers,
            )

        if any(num != 9 for num in counter.values()):
            raise ValueError(
                "There is an uneven distribution of sides (should be exactly 9 for each side).",
                facelets,
                counter,
            )

        # convert the text notation to a numerical equivalent then construct a cubie cube from it
        face_indexes = [centers.index(facelet) for facelet in facelets]
        util.to_cubie_cube(face_indexes, self.cc)
        self.cc.verify()

    def _solve(self) -> str:
        """
        private String solve(CubieCube c) {
            Tools.init();
            int conjMask = 0;
            for (int i=0; i<6; i++) {
                twist[i] = c.getTwistSym();
                flip[i] = c.getFlipSym();
                slice[i] = c.getUDSlice();
                corn0[i] = c.getCPermSym();
                ud8e0[i] = c.getU4Comb() << 16 | c.getD4Comb();

                for (int j=0; j<i; j++) {
                    //If S_i^-1 * C * S_i == C, It's unnecessary to compute it again.
                    if (twist[i] == twist[j] && flip[i] == flip[j] && slice[i] == slice[j]
                            && corn0[i] == corn0[j] && ud8e0[i] == ud8e0[j]) {
                        conjMask |= 1 << i;
                        break;
                    }
                }
                if ((conjMask & (1 << i)) == 0) {
                    prun[i] = Math.max(Math.max(
                        CoordCube.getPruning(CoordCube.UDSliceTwistPrun,
                            (twist[i]>>>3) * 495 +
                                CoordCube.UDSliceConj[slice[i]&0x1ff][twist[i]&7]),
                        CoordCube.getPruning(CoordCube.UDSliceFlipPrun,
                            (flip[i]>>>3) * 495 +
                                CoordCube.UDSliceConj[slice[i]&0x1ff][flip[i]&7])),
                        Tools.USE_TWIST_FLIP_PRUN ? CoordCube.getPruning(CoordCube.TwistFlipPrun,
                                (twist[i]>>>3) * 2688 + (flip[i] & 0xfff8 |
                                    CubieCube.Sym8MultInv[flip[i]&7][twist[i]&7])) : 0);
                }
                c.URFConjugate();
                if (i==2) {
                    c.invCubieCube();
                }
            }
            for (depth1=0; depth1<sol; depth1++) {
                maxDep2 = Math.min(12, sol-depth1);
                for (urfIdx=0; urfIdx<6; urfIdx++) {
                    if((firstAxisRestriction != -1 || lastAxisRestriction != -1) && urfIdx >= 3) {
                        // When urfIdx >= 3, we're solving the
                        // inverse cube. This doesn't work
                        // when we're also restricting the
                        // first turn, so we just skip inverse
                        // solutions when firstAxisRestriction has
                        // been set.
                        continue;
                    }
                    if ((conjMask & (1 << urfIdx)) != 0) {
                        continue;
                    }
                    corn[0] = corn0[urfIdx];
                    mid4[0] = slice[urfIdx];
                    ud8e[0] = ud8e0[urfIdx];
                    valid1 = 0;
                    int lm = firstAxisRestriction == -1 ? -1 :
                        CubieCube.urfMoveInv[urfIdx][firstAxisRestriction]/3*3;
                    if ((prun[urfIdx] <= depth1)
                            && phase1(twist[urfIdx]>>>3, twist[urfIdx]&7,
                                flip[urfIdx]>>>3, flip[urfIdx]&7,
                                slice[urfIdx]&0x1ff, depth1, lm) == 0) {
                        return solution == null ? "Error 8" : solution;
                    }
                }
            }
            return solution == null ? "Error 7" : solution;
        }
        """
        conj_mask = 0  # conjugation mask bits
        conj_mask2 = [False] * 6
        conj_checks = [self.twist, self.flip, self.slice, self.corn0, self.ud8e0]

        for i in range(6):
            self.twist[i] = self.cc.get_twist_sym()
            self.flip[i] = self.cc.get_flip_sym()
            self.slice[i] = self.cc.get_ud_slice()
            self.corn0[i] = self.cc.get_c_perm_sym()
            self.ud8e0[i] = self.cc.get_u4_comb() << 16 | self.cc.get_d4_comb()

            for j in range(i):
                if all(c[i] == c[j] for c in conj_checks):
                    # If S_i^-1 * C * S_i == C, It's unnecessary to compute it again.
                    conj_mask |= 1 << i
                    conj_mask2[i] = True
                    break

            # if conj_mask & (1 << i) == 0:
            if not conj_mask2[i]:
                slice_twist_prune = coord_cube.get_pruning(
                    coord_cube.UD_SLICE_TWIST_PRUNE,
                    (self.twist[i] >> 3) * 495 +
                    coord_cube.UD_SLICE_CONJUGATE[self.slice[i] & 0x1ff][self.twist[i] & 7],
                )

                slice_flip_prune = coord_cube.get_pruning(
                    coord_cube.UD_SLICE_FLIP_PRUNE,
                    (self.flip[i] >> 3) * 495 +
                    coord_cube.UD_SLICE_CONJUGATE[self.slice[i] & 0x1ff][self.flip[i] & 7],
                )

                twist_flip_prune = 0

                if tools.USE_TWIST_FLIP_PRUNE:
                    twist_flip_prune = coord_cube.get_pruning(
                        coord_cube.TWIST_FLIP_PRUNE,
                        (self.twist[i] >> 3) * 2688 + (
                            self.flip[i] & 0xfff8 |
                            cubie_cube.SYM_8_MULT_INV[self.flip[i] & 7][self.twist[i] & 7]
                        ),
                    )

                self.prune[i] = max(slice_twist_prune, slice_flip_prune, twist_flip_prune)

            self.cc.urf_conjugate()
            if i == 2:
                self.cc.inv_cubie_cube()

        for depth1 in range(min(self.prune), self.sol):
            self.depth1 = depth1
            self.max_depth2 = min(12, self.sol - depth1)
            for urf_idx in range(6):
                if conj_mask2[urf_idx]:
                    continue

                self.urf_idx = urf_idx
                if ((self.first_axis_restriction is not None or
                        self.last_axis_restriction is not None) and
                        urf_idx >= 3):
                    # When urfIdx >= 3, we're solving the
                    # inverse cube. This doesn't work
                    # when we're also restricting the
                    # first turn, so we just skip inverse
                    # solutions when firstAxisRestriction has
                    # been set.
                    continue

                self.corn[0] = self.corn0[urf_idx]
                self.mid4[0] = self.slice[urf_idx]
                self.ud8e[0] = self.ud8e0[urf_idx]
                self.valid1 = 0
                lm = (
                    -1 if self.first_axis_restriction is None else
                    cubie_cube.URF_MOVE_INV[urf_idx][self.first_axis_restriction]
                )
                # if self.prune[urf_idx] <= depth1:
                #     print("Calling outer Phase 1")
                if self.prune[urf_idx] <= depth1:
                    print("calling phase1 from solve",
                          self.move_set_count, self.move, self.depth1, self.urf_idx)
                    x = self._phase1(
                        twist=self.twist[urf_idx] >> 3,
                        twist_sym=self.twist[urf_idx] & 7,
                        flip=self.flip[urf_idx] >> 3,
                        flip_sym=self.flip[urf_idx] & 7,
                        ud_slice=self.slice[urf_idx] & 0x1ff,
                        max_moves=depth1,
                        last_axis=lm,
                    )
                    if x == 0:
                        if self.solution is None:
                            raise ValueError("Error 8: Timeout, no solution within given time")
                        return self.solution
        if self.solution is None:
            raise ValueError("Error 7: No solution exists for the given maxDepth")
        return self.solution

    def _phase1(
        self,
        twist: int,
        twist_sym: int,
        flip: int,
        flip_sym: int,
        ud_slice: int,
        max_moves: int,
        last_axis: int,
    ):
        """
        In phase 1, the algorithm looks for maneuvers which will transform a scrambled cube to
        G1. That is, the orientations of corners and edges have to be constrained and the edges
        of the UD-slice have to be transferred into that slice. In phase 2 we restore the cube.

        If you turn the faces of a solved cube and do not use the moves R, R', L, L', F, F',
        B and B' you will only generate a subset of all possible cubes. This subset is denoted
        by G1 = <U,D,R2,L2,F2,B2>.

        There are many different possibilities for maneuvers in phase 1. The algorithm tries
        different phase 1 maneuvers to find a most possible short overall solution.

        In phase 1, any cube is described with three coordinates:
        The corner orientation coordinate(0..2186), the edge orientation coordinate (0..2047)
        and UDSlice coordinate. The UDSlice coordinate is number from 0 to 494
        (12*11*10*9/4! - 1) which is determined by the positions of the 4 UDSlice edges.
        The order of the 4 UDSlice edges within the positions is ignored.

        The problem space of phase 1 has 2187*2048*495 = 2.217.093.120

        @return
            0: Found or Timeout
            1: Try Next Power
            2: Try Next Axis
        """
        print("\nPhase 1", twist, flip, ud_slice, max_moves, end='\n')
        if twist == 0 and flip == 0 and ud_slice == 0 and max_moves < 5:
            print("early return", max_moves)
            return self._init_phase2() if max_moves == 0 else 1

        for axis in range(0, 18, 3):
            if axis == last_axis or axis == last_axis - 9:
                continue

            for power in range(3):
                m = axis + power

                slice_x = coord_cube.UD_SLICE_MOVE[ud_slice][m] & 0x1ff
                twist_x = coord_cube.TWIST_MOVE[twist][cubie_cube.SYM_8_MOVE[twist_sym][m]]
                t_sym_x = cubie_cube.SYM_8_MULT[twist_x & 7][twist_sym]
                if twist_x < 0:
                    raise ValueError(twist_x)
                twist_x >>= 3
                prune = coord_cube.get_pruning(
                    coord_cube.UD_SLICE_TWIST_PRUNE,
                    twist_x * 495 + coord_cube.UD_SLICE_CONJUGATE[slice_x][t_sym_x],
                )

                if prune > max_moves:
                    break

                if prune == max_moves:
                    continue

                flip_x = coord_cube.FLIP_MOVE[flip][cubie_cube.SYM_8_MOVE[flip_sym][m]]
                f_sym_x = cubie_cube.SYM_8_MULT[flip_x & 7][flip_sym]
                if flip_x < 0:
                    raise ValueError(flip_x)
                flip_x >>= 3

                if tools.USE_TWIST_FLIP_PRUNE:
                    prune = coord_cube.get_pruning(
                        coord_cube.TWIST_FLIP_PRUNE,
                        (twist_x * 336 + flip_x) << 3 |
                        cubie_cube.SYM_8_MULT_INV[f_sym_x][t_sym_x],
                    )

                    if prune > max_moves:
                        break

                    if prune == max_moves:
                        continue

                prune = coord_cube.get_pruning(
                    coord_cube.UD_SLICE_FLIP_PRUNE,
                    flip_x * 495 + coord_cube.UD_SLICE_CONJUGATE[slice_x][f_sym_x]
                )

                if prune > max_moves:
                    break

                if prune == max_moves:
                    continue

                self.move[self.depth1 - max_moves] = m
                self.phase1_move[self.depth1 - max_moves] = m
                self.move_set_count += 1

                if self.move_set_count > 65000:
                    raise ValueError('hello')
                print("Calling recursive Phase 1")
                ret = self._phase1(
                    twist=twist_x,
                    twist_sym=t_sym_x,
                    flip=flip_x,
                    flip_sym=f_sym_x,
                    ud_slice=slice_x,
                    max_moves=max_moves - 1,
                    last_axis=axis,
                )

                if ret != 1:
                    return ret >> 1
        return 1

    def _init_phase2(self):
        """
        In phase 2, any cube is also described with three coordinates:

        The corner permutation coordinate (0..40319), the phase 2 edge permutation coordinate (
        0..40319), and the phase2 UDSlice coordinate (0..23).

        We have 8! = 40320 possibilities to permute the 8 edges of the U and D face (remember
        that we only allow 180 degree turns for all faces R, L, F and B).

        The phase 2 triple (0,0,0) belongs to a pristine cube.

        The problem space of phase 2 has 40320*40320*24/2 = 19.508.428.800 different states.

        @return
            0: Found or Timeout
            1: Try Next Power
            2: Try Next Axis
        """
        # print("\nPhase 2 init")
        if time.time() >= (self.time_out if self.solution is None else self.time_min):
            print("time out")
            return 0

        # print(self.valid1, self.valid2)
        self.valid2 = min(self.valid1, self.valid2)
        if self.corn[self.valid1] < 0:
            raise ValueError(self.corn[self.valid1])
        c_idx = self.corn[self.valid1] >> 4
        c_sym = self.corn[self.valid1] & 0xf

        for i in range(self.valid1, self.depth1):
            m = self.move[i]
            c_idx = coord_cube.C_PERM_MOVE[c_idx][cubie_cube.SYM_MOVE[c_sym][m]]
            c_sym = cubie_cube.SYM_MULT[c_idx & 0xf][c_sym]
            if c_idx < 0:
                raise ValueError(c_idx)
            c_idx >>= 4
            self.corn[i + 1] = c_idx << 4 | c_sym

            cx = coord_cube.UD_SLICE_MOVE[self.mid4[i] & 0x1ff][m]
            if self.mid4[i] < 0 or cx < 0:
                raise ValueError(self.mid4[i], cx)
            self.mid4[i + 1] = util.PERM_MULT[self.mid4[i] >> 9][cx >> 9] << 9 | cx & 0x1ff

        self.valid1 = self.depth1
        if self.mid4[self.depth1] < 0:
            raise ValueError(self.mid4[self.depth1])
        mid = self.mid4[self.depth1] >> 9
        prune = coord_cube.get_pruning(
            coord_cube.MC_PERM_PRUNE,
            c_idx * 24 + coord_cube.M_PERM_CONJUGATE[mid][c_sym],
        )

        if prune >= self.max_depth2:
            return 2 if prune > self.max_depth2 else 1

        if self.ud8e[self.valid2] < 0:
            raise ValueError(self.ud8e[self.valid2])
        u4e = self.ud8e[self.valid2] >> 16 & 0xffff
        d4e = self.ud8e[self.valid2] & 0xffff

        for i in range(self.valid2, self.depth1):
            m = self.move[i]

            cx = coord_cube.UD_SLICE_MOVE[u4e & 0x1ff][m]
            u4e = util.PERM_MULT[d4e >> 9][cx >> 9] << 9 | cx & 0x1ff

            cx = coord_cube.UD_SLICE_MOVE[d4e & 0x1ff][m]
            d4e = util.PERM_MULT[d4e >> 9][cx >> 9] << 9 | cx & 0x1ff

            self.ud8e[i + 1] = u4e << 16 | d4e

        self.valid2 = self.depth1

        edge_idx = cubie_cube.M2E_PERM[494 - (u4e & 0x1ff) + (u4e >> 9) * 70 + (d4e >> 9) * 1680]
        e_sym = edge_idx & 15
        edge_idx >>= 4

        prune = max(
            coord_cube.get_pruning(
                coord_cube.ME_PERM_PRUNE, edge_idx * 24 + coord_cube.M_PERM_CONJUGATE[mid][e_sym]
            ),
            prune,
        )

        if prune >= self.max_depth2:
            return 2 if prune > self.max_depth2 else 1

        if self.depth1 != 0:
            lm = util.STD_2_UD[self.move[self.depth1 - 1] // 3 * 3 + 1]
        elif self.first_axis_restriction is not None:
            lm = util.STD_2_UD[
                cubie_cube.URF_MOVE_INV[self.urf_idx][self.first_axis_restriction] // 3 * 3 + 1
            ]
        else:
            lm = 10

        for depth2 in range(prune, self.max_depth2):
            print("Calling phase2 from phase2 init", depth2)
            ret = self._phase2(
                edge_idx=edge_idx,
                edge_sym=e_sym,
                corner_idx=c_idx,
                corner_sym=c_sym,
                mid=mid,
                max_moves=depth2,
                depth=self.depth1,
                lm=lm,
            )

            if ret:
                self.sol = self.depth1 + depth2
                self.max_depth2 = min(12, self.sol - self.depth1)
                self.solution = self._solution_to_string()
                print("checking min time", time.time() >= self.time_min)
                return 0
                return 0 if time.time() >= self.time_min else 1
        return 1

    def _phase2(
        self,
        edge_idx: int,
        edge_sym: int,
        corner_idx: int,
        corner_sym: int,
        mid: int,
        max_moves: int,
        depth: int,
        lm: int,
    ):
        if max_moves == 0:
            # We've done the last move we're allowed to do, make sure
            # it's permitted by lastAxisRestriction.
            if self.last_axis_restriction is not None:
                std_lm = cubie_cube.URF_MOVE[self.urf_idx][util.UD_2_STD[lm]]
                last_axis = (std_lm // 3) * 3
                if (self.last_axis_restriction == last_axis or
                        self.last_axis_restriction == last_axis + 9):
                    return False

            # Is the cube solved?
            return edge_idx == 0 and corner_idx == 0 and mid == 0

        for m in range(10):
            if util.CKMV2[lm][m]:
                continue

            mid_x = coord_cube.M_PERM_MOVE[mid][m]
            c_idx_x = coord_cube.C_PERM_MOVE[corner_idx][
                cubie_cube.SYM_MOVE[corner_sym][util.UD_2_STD[m]]
            ]
            c_sym_x = cubie_cube.SYM_MULT[c_idx_x & 15][corner_sym]
            c_idx_x >>= 4

            prune = coord_cube.get_pruning(
                coord_cube.MC_PERM_PRUNE,
                c_idx_x * 24 + coord_cube.M_PERM_CONJUGATE[mid_x][c_sym_x],
            )

            if prune >= max_moves:
                continue

            e_idx_x = coord_cube.E_PERM_MOVE[edge_idx][cubie_cube.SYM_MOVE_UD[edge_sym][m]]
            e_sym_x = cubie_cube.SYM_MULT[e_idx_x & 15][edge_sym]
            e_idx_x >>= 4

            prune = coord_cube.get_pruning(
                coord_cube.ME_PERM_PRUNE,
                e_idx_x * 24 + coord_cube.M_PERM_CONJUGATE[mid_x][e_sym_x],
            )

            if prune >= max_moves:
                continue

            ret = self._phase2(
                edge_idx=e_idx_x,
                edge_sym=e_sym_x,
                corner_idx=c_idx_x,
                corner_sym=c_sym_x,
                mid=mid_x,
                max_moves=max_moves - 1,
                depth=depth + 1,
                lm=m,
            )

            if ret:
                self.move[depth] = util.UD_2_STD[m]
                self.phase2_move[depth] = util.UD_2_STD[m]
                print("found phase 2 result", depth, util.UD_2_STD[m], m, self.phase2_move)
                return True

        return False

    def _solution_to_string(self) -> str:
        """
        private String solutionToString() {
            StringBuffer sb = new StringBuffer();
            int urf = (verbose & INVERSE_SOLUTION) != 0 ? (urfIdx + 3) % 6 : urfIdx;
            if (urf < 3) {
                for (int s=0; s<depth1; s++) {
                    sb.append(Util.move2str[CubieCube.urfMove[urf][move[s]]]).append(' ');
                }
                if ((verbose & USE_SEPARATOR) != 0) {
                    sb.append(".  ");
                }
                for (int s=depth1; s<sol; s++) {
                    sb.append(Util.move2str[CubieCube.urfMove[urf][move[s]]]).append(' ');
                }
            } else {
                for (int s=sol-1; s>=depth1; s--) {
                    sb.append(Util.move2str[CubieCube.urfMove[urf][move[s]]]).append(' ');
                }
                if ((verbose & USE_SEPARATOR) != 0) {
                    sb.append(".  ");
                }
                for (int s=depth1-1; s>=0; s--) {
                    sb.append(Util.move2str[CubieCube.urfMove[urf][move[s]]]).append(' ');
                }
            }
            if ((verbose & APPEND_LENGTH) != 0) {
                sb.append("(").append(sol).append("f)");
            }
            return sb.toString();
        }
        """
        urf = (
            (self.urf_idx + 3) % 6 if self.inverse_solution else self.urf_idx
        )
        separator = '.  ' if self.use_separator != 0 else ''
        len_post = f' ({self.sol}f)' if self.append_length != 0 else ''

        if urf < 3:
            phase_1 = ' '.join(
                util.MOVE_2_STR[cubie_cube.URF_MOVE[urf][self.move[s]]]
                for s in range(self.depth1)
            )
            phase_2 = ' '.join(
                util.MOVE_2_STR[cubie_cube.URF_MOVE[urf][self.move[s]]]
                for s in range(self.depth1, self.sol)
            )
        else:
            phase_1 = ' '.join(
                util.MOVE_2_STR[cubie_cube.URF_MOVE[urf][self.move[s]]]
                for s in range(self.sol - 1, self.depth1 - 1, -1)
            )
            phase_2 = ' '.join(
                util.MOVE_2_STR[cubie_cube.URF_MOVE[urf][self.move[s]]]
                for s in range(self.depth1 - 1, -1, -1)
            )

        return f'{phase_1} {separator}{phase_2}{len_post}'
