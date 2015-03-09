/*本程序用于提取图像的前景轮廓。本项目组的工作是将原先的OpenCV 1.X代码修改为OpenCV 2.X代码。
 * 代码在WIN7 + Qt5.2.0 + OpenCV2.4.8平台上测试，成功运行。
 * Tips: input文件夹、trimap文件夹与result文件夹放在程序运行的当前目录下，即可正常运行，产生结果。
 * 2014年1月7日
*/
#include "sharedmatting.h"
#include <time.h>
#include <string>

using namespace std;

int main()
{
    double sum = 0;;
    char fileAddr[64] = {0};

    int n;
    for (n = 1; n < 28; ++n) {
        SharedMatting sm;
        clock_t start, finish;
        start = clock();
        sprintf(fileAddr, "input/GT%d%d.png", n/10, n%10);
        sm.loadImage(fileAddr);
        sprintf(fileAddr, "trimap/trimap1/GT%d%d.png", n/10, n%10);
        sm.loadTrimap(fileAddr);
        sm.solveAlpha();
        sprintf(fileAddr, "result/GT%d%d.png", n/10, n%10);
        sm.save(fileAddr);
        finish = clock();
        cout <<  "Time used by processing GT" << n/10 << n%10 << ".png: "
             << double(finish - start) / CLOCKS_PER_SEC << endl;
        sum += double(finish - start) / CLOCKS_PER_SEC;
    }
    cout << "Time used per image( 27 images): " << sum / 27 << endl;

    return 0;
}
