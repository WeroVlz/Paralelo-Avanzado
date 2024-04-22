#include <iostream>
#include <vector>
#include <ctime>
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <string>

#define N 9

__global__ void verifySudokuKernel(const int* board, bool* result) {
    int idx = threadIdx.x;
    if (idx >= N) return;

    //ROWS
    bool checked_row[N] = { false };
    for (int j = 0; j < N; ++j) {
        int num = board[idx * N + j];
        if (num != 0) {
            if (checked_row[num - 1]) {
                result[idx] = false;
                return;
            }
            checked_row[num - 1] = true;
        }
    }

    //COLUMNS
    bool check_column[N] = { false };
    for (int i = 0; i < N; ++i) {
        int num = board[i * N + idx];
        if (num != 0) {
            if (check_column[num - 1]) {
                result[idx] = false;
                return;
            }
            check_column[num - 1] = true;
        }
    }

    int rowBlock = (idx / 3) * 3;
    int columnBlock = (idx % 3) * 3;
    bool check_block[N] = { false };
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            int num = board[(rowBlock + i) * N + (columnBlock + j)];
            if (num != 0) {
                if (check_block[num - 1]) {
                    result[idx] = false;
                    return;
                }
                check_block[num - 1] = true;
            }
        }
    }
    result[idx] = true;
}

bool verifySudokuValidity(const std::vector<std::vector<int>>& board) {
    int* d_board;
    bool* d_result;
    bool h_result[N];

    cudaMalloc(&d_board, N * N * sizeof(int));
    cudaMalloc(&d_result, N * sizeof(bool));

    int tablero_plano[N * N];
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            tablero_plano[i * N + j] = board[i][j];
        }
    }

    cudaMemcpy(d_board, tablero_plano, N * N * sizeof(int), cudaMemcpyHostToDevice);

    verifySudokuKernel << <1, N >> > (d_board, d_result);

    cudaMemcpy(h_result, d_result, N * sizeof(bool), cudaMemcpyDeviceToHost);

    cudaFree(d_board);
    cudaFree(d_result);

    for (int i = 0; i < N; ++i) {
        if (!h_result[i]) {
            return false;
        }
    }

    return true;
}

bool solveSudoku(std::vector<std::vector<int>>& board, int row = 0, int column = 0) {

    while (row < 9 && board[row][column] != 0) {
        column++;
        if (column == 9) {
            column = 0;
            row++;
        }
    }
    if (row == 9) {
        return true;
    }

    for (int num = 1; num <= 9; ++num) {
        bool isValid = true;
        for (int i = 0; i < N; ++i) {
            if (board[row][i] == num) {
                isValid = false;
                break;
            }
        }

        for (int i = 0; i < N; ++i) {
            if (board[i][column] == num) {
                isValid = false;
                break;
            }
        }

        int startRow = (row / 3) * 3;
        int startColumn = (column / 3) * 3;
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                if (board[startRow + i][startColumn + j] == num) {
                    isValid = false;
                    break;
                }
            }
        }

        if (isValid) {
            board[row][column] = num;
            if (solveSudoku(board, row, column)) {
                return true;
            }
            board[row][column] = 0;
        }
    }

    return false;
}

void printSudokuBoard(const std::vector<std::vector<int>>& board) {

    const std::string line = "+-------+-------+-------+";

    std::cout << line << std::endl;

    for (int i = 0; i < N; ++i) {
        std::cout << "| ";
        for (int j = 0; j < N; ++j) {
            std::cout << (board[i][j] == 0 ? "." : std::to_string(board[i][j])) << " ";
            if (j % 3 == 2) {
                std::cout << "| ";
            }
        }

        std::cout << std::endl;
        if (i % 3 == 2) {
            std::cout << line << std::endl;
        }
    }
}

int main() {
    std::vector<std::vector<int>> board = {
        {9, 1, 3, 4, 2, 7, 0, 8, 0},
        {6, 0, 0, 0, 0, 0, 0, 0, 0},
        {2, 0, 0, 0, 0, 3, 0, 7, 0},
        {0, 0, 0, 1, 0, 2, 0, 0, 8},
        {0, 6, 2, 5, 0, 0, 0, 0, 3},
        {5, 3, 8, 7, 0, 0, 2, 9, 0},
        {3, 4, 0, 8, 7, 0, 0, 6, 0},
        {0, 0, 6, 0, 4, 9, 8, 1, 5},
        {8, 0, 1, 2, 0, 0, 0, 0, 0}
    };

    if (!verifySudokuValidity(board)) {
        std::cout << "The Sudoku board is not valid.\n" << std::endl;
        return 1;
    }

    clock_t startClock = clock();

    if (solveSudoku(board)) {
        clock_t endClock = clock();
        double time = double(endClock - startClock) / CLOCKS_PER_SEC;

        std::cout << "Sudoku solved in " << time << " seconds.\n" << std::endl;
        printSudokuBoard(board);
    }
    else {
        std::cout << "The Sudoku could not be solved.\n" << std::endl;
    }

    return 0;
}