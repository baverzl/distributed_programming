#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <algorithm>
#include <cctype>
#include <cctype>
#include <locale>

using namespace std;

// trim from start (in place)
static inline void ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
}
// trim from end (in place)
static inline void rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(),
                std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
}

bool isPalindrome(string &s)
{
    int n = s.length();
    for(int i = 0; i < n/2; i++) {
        if(s[i] != s[n - i - 1])
            return false;
    }
    return true;
}
string &reverse(string s)
{
    string &res = *(new string(s));
    int n = s.length();
    int tmp;
    for(int i = 0; i < n / 2; i++) {
        tmp = res[i];
        res[i] = res[n - i - 1];
        res[n - i - 1] = tmp;
    }
    return res;
}

int main(int argc, char *argv[])
{
    if( argc != 2 ) { 
        printf("Usage: %s <thread_count>\n", argv[0]);
        return -1;
    }

    int thread_count = atoi(argv[1]);

    double start, end;

    start = omp_get_wtime();

    string word;
    vector<string> words;

    ifstream inFile("words.txt");
    ofstream outFile("results.txt");

    while(!inFile.eof()) {
        getline(inFile, word);
        rtrim(word);
        words.push_back(word);
    }

    inFile.close();

#pragma omp parallel for ordered schedule(dynamic) num_threads(thread_count) shared(words)
    for(int i = 0; i < words.size(); i++) {
            if(isPalindrome(words[i]) || find(words.begin(), words.end(), reverse(words[i])) != words.end() ) {
#pragma omp ordered
                outFile << words[i] << endl;
                
            }
    }

    outFile.close();

    end = omp_get_wtime();

    cout << end - start << endl;

    return 0;
}
