#include <iostream>
#include <thread>
#include <chrono>

using namespace std;
using namespace std::chrono;

unsigned long long N = 10000000000;
unsigned long long sum = 0;

const int MAXTHREADS = 16;

unsigned long long sumArr[MAXTHREADS]{};

void Calculate(unsigned long long begin, unsigned long long end, int thread_n) {
	unsigned long long sum = 0;
	for (auto i = begin; i < end; i++)  sum += i;

	sumArr[thread_n] += sum;
}

int main() {
	// без потоков

	auto start1 = high_resolution_clock::now();

	for (unsigned long long i = 1; i <= N; i++) {
		sum = sum + i;
	}

	auto end1 = high_resolution_clock::now();

	auto duration1 = duration_cast<milliseconds>(end1 - start1);
	cout << "No thread sum:" << sum << endl;

	cout << "Time: " << duration1.count() << " milliseconds\n" << endl;


	//с потоками
	auto start2 = high_resolution_clock::now();

	thread threads[MAXTHREADS];

	for (int i = 0; i < MAXTHREADS; i++) 
	{
		if (i == 0) 
		{
			threads[i] = thread(Calculate, 0, N / MAXTHREADS, i);
		}
		else 
		{
			threads[i] = thread(Calculate, ((N * i) / MAXTHREADS), ((i + 1) * N) / MAXTHREADS, i);
		}
	}

	for (auto i = 0; i < MAXTHREADS; i++) {
		threads[i].join();
	}

	auto end2 = high_resolution_clock::now();
	auto duration2 = duration_cast<milliseconds>(end2 - start2);
	unsigned long long threadsum = 0;
	for (int i = 0; i < MAXTHREADS; i++)
	{
		threadsum += sumArr[i];
	}
	cout << "thread sum:" << threadsum << endl;
	cout << "Time: " << duration2.count() << " milliseconds\n" << endl;

	return 0;
}