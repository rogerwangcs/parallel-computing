// Author: Roger Wang
// Modified Game of Life on CPU by Daniel Angel Jimenez

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define GAME_DIM_X 20
#define GAME_DIM_Y 20

void createBoard(int board[][GAME_DIM_Y]) {
    int i, j;

    for (i = 0; i < GAME_DIM_X; i++)
        for (j = 0; j < GAME_DIM_Y; j++)
            board[i][j] = rand() % 2;  // either 0 or 1
}

int addX(int i, int a) {
    i += a;
    while (i < 0) i += GAME_DIM_X;
    while (i >= GAME_DIM_X) i -= GAME_DIM_X;
    return i;
}

int addY(int i, int a) {
    i += a;
    while (i < 0) i += GAME_DIM_Y;
    while (i >= GAME_DIM_Y) i -= GAME_DIM_Y;
    return i;
}

int adjacentCells(int board[][GAME_DIM_Y], int i, int j) {
    int count = 0;

    for (int k = -1; k <= 1; k++)
        for (int l = -1; l <= 1; l++)
            if (k || l)
                if (board[addX(i, k)][addY(j, l)]) count++;
    return count;
}

void play(int board[][GAME_DIM_Y]) {
    int newboard[GAME_DIM_X][GAME_DIM_Y];
    int a;

    for (int i = 0; i < GAME_DIM_X; i++) {
        for (int j = 0; j < GAME_DIM_Y; j++) {
            int adj = adjacentCells(board, i, j);
            if (adj == 2) newboard[i][j] = board[i][j];
            if (adj == 3) newboard[i][j] = 1;
            if (adj < 2) newboard[i][j] = 0;
            if (adj > 3) newboard[i][j] = 0;
        }
    }

    for (int i = 0; i < GAME_DIM_X; i++) {
        for (int j = 0; j < GAME_DIM_Y; j++) {
            board[i][j] = newboard[i][j];
        }
    }
}

void showBoard(int board[][GAME_DIM_Y]) {
    for (int j = 0; j < GAME_DIM_Y; j++) {
        for (int i = 0; i < GAME_DIM_X; i++) {
            if (i == 0)
                printf("║");
            printf("%s", board[i][j] ? "██" : "  ");
            if (i == GAME_DIM_X - 1)
                printf("║");
        }
        printf("\n");
    }
    for (int i = 0; i < GAME_DIM_X - 1 + 2; i++)
        printf("──");
    printf("\n");
}

int main(int argc, char *argv[]) {
    int gameBoard[GAME_DIM_X][GAME_DIM_Y];

    createBoard(gameBoard);
    printf("%d", argc);

    while (1) {
        showBoard(gameBoard);
        play(gameBoard);
        usleep(1000 * 50);
    }
}