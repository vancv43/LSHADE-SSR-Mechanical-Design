/*
 * L-SHADE-SSR: A Hybrid Success-History Differential Evolution with Staggered Sub-population Restarts
 * Authors: Chau Van Van (1,3,*), Nguyen Minh Tuan (1), Chau Khai Kien (2), and Phayung Meesad (3,*)
 * (1) Faculty of Information Technology, Posts and Telecommunications Institute of Technology,
 *     11 Nguyen Dinh Chieu Street, Sai Gon Ward, Ho Chi Minh City 700000, Vietnam
 *     vancv@ptit.edu.vn, minhtuan@ptit.edu.vn
 * (2) Faculty of Artificial Intelligence, Asia University Viet Nam,
 *     No. 485 Le Quang Dinh, Ho Chi Minh City, Vietnam
 *     khaikien0763@gmail.com
 * (3) Department of Information Technology Management, Faculty of Information Technology and Digital Innovation,
 *     King Mongkut's University of Technology North Bangkok,
 *     1518 Pracharat 1 Rd, Wongsawang, Bangsue, Bangkok, 10800, Thailand
 *     phayung.m@itd.kmutnb.ac.th
 * (*) Corresponding authors
 * Repository: https://github.com/vancv43/LSHADE-SSR-Mechanical-Design (MIT License)
 */
// Thin C wrapper around the unmodified official CEC2017 code (P-N-Suganthan/CEC2017-BoundContrained).
#include <stdlib.h>
double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
#include "cec17_test_func.cpp"
extern "C" void cec17(double *x, double *f, int nx, int mx, int func_num) { cec17_test_func(x, f, nx, mx, func_num); }
