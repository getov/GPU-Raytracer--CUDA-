#include "Util.cuh"
#include <string>
using std::string;

std::vector<string> tokenize(string s)
{
	int i = 0, j, l = (int) s.length();
	std::vector<string> result;
	while (i < l) {
		while (i < l && isspace(s[i])) i++;
		if (i >= l) break;
		j = i;
		while (j < l && !isspace(s[j])) j++;
		result.push_back(s.substr(i, j - i));
		i = j;
	}
	return result;
}

std::vector<string> split(string s, char separator)
{
	int i = 0, j, l = (int) s.length();
	std::vector<string> result;
	while (i < l) {
		j = i;
		while (j < l && s[j] != separator) j++;
		result.push_back(s.substr(i, j - i));
		i = j + 1;
		if (j == l - 1) result.push_back("");
	}
	return result;
}

//__device__
//c_string::c_string()
//	: temp(nullptr)
//{
//}
//
//__device__
//c_string::~c_string()
//{
//	delete [] temp;
//}
//
//__device__
//int c_string::length(const char* s)
//{
//	int count = 0;
//
//	while (*(s++)) ++count;
//
//	return count;
//}
//
//__device__
//bool c_string::isspace(const char s)
//{
//	if (s == ' ' ||
//		s == '\t' ||
//		s == '\n' ||
//		s == '\v' ||
//		s == '\f' ||
//		s == '\r')
//	{
//		return true;
//	}
//	else
//	{
//		return false;
//	}
//}
//
//__device__
//char* c_string::substr(const char* s, int pos, int length)
//{
//	temp = new char[length + 1];
//
//	int i;
//	s += pos;
//
//	for (i = 0; i < length; ++i)
//	{
//		*(temp + i) = *s;
//		++s;
//	}
//
//	*(temp + i) = '\0';
//
//	return temp;
//}
//
//__device__
//int c_string::atoi(const char* s)
//{
//	int num = 0;
//	while(*s)
//	{
//		num = ((*s) - '0') + num * 10;
//		s++;   
//	}
//	return num;
//}
//
//
//__device__
//vector<char*> tokenize(char* s)
//{
//	c_string str;
//
//	int i = 0, j, l = str.length(s);
//	vector<char*> result;
//	while (i < l) {
//		while (i < l && str.isspace(s[i])) i++;
//		if (i >= l) break;
//		j = i;
//		while (j < l && !str.isspace(s[j])) j++;
//		result.push_back(str.substr(s, i, j - i));
//		i = j;
//	}
//	return result;
//}
//
//__device__
//vector<char*> split(const char* s, char separator)
//{
//	c_string str;
//
//	int i = 0, j, l = str.length(s);
//	vector<char*> result;
//	while (i < l) {
//		j = i;
//		while (j < l && s[j] != separator) j++;
//		result.push_back(str.substr(s, i, j - i));
//		i = j + 1;
//		if (j == l - 1) result.push_back("");
//	}
//	return result;
//}