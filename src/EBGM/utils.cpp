#include "utils.h"

#include <cmath>
#include <tuple>
#include <iostream>
#include <exception>
#include <regex>

#include <boost/filesystem.hpp>

using namespace std;

ostream *Log::os = &std::clog;


// if path doesn't exist, return false
// if path is a zero-length file, return false
// if path is a directory, return false
bool fileexists(const boost::filesystem::path &path)
{
	using namespace boost::filesystem;

	//status: follow symbol links
	//symlink_status: don't follow symbol links.
	file_type type = status(path).type();

	if (type == file_not_found || type == directory_file || type == status_error) return false;

	if (type == regular_file && file_size(path) == 0) return false;

	return true;
}

bool extensionIs(const std::string &filename, const std::string &ext)
{
	int dotPos = filename.find_last_of('.');
	return !filename.substr(dotPos, ext.size()).compare(ext);

}

// if path is a file, return its parent directory.
// if path is a directory, return itself.
// if neither, throw a runtime_error.
boost::filesystem::path getDirectory(const boost::filesystem::path &path)
{
	using namespace boost::filesystem;

	boost::filesystem::path ret;

	//status: follow symbol links
	//symlink_status: don't follow symbol links.
	file_type type = status(path).type();

	if(type == regular_file){
		ret = path.parent_path();
		return ret;
	}

	if(type == directory_file){
		ret = path;
		return ret;
	}

	throw(runtime_error(
		string("getDirectory(): '") +
		path.string() +
		"' is neither a file nor a directory."
	));

}

// check the syntax of a regular expression
bool checkregex(const char *regex, std::string &errordescription) noexcept
{
	try {
		std::regex re(regex);
	}
	catch (std::regex_error ex) {
		std::regex_constants::error_type code = ex.code();
		switch (code)
		{
		case std::regex_constants::error_collate:
			errordescription.assign("The expression contained an invalid collating element name.");
			break;
		case std::regex_constants::error_ctype:
			errordescription.assign("The expression contained an invalid character class name.");
			break;
		case std::regex_constants::error_escape:
			errordescription.assign("The expression contained an invalid escaped character, or a trailing escape.");
			break;
		case std::regex_constants::error_backref:
			errordescription.assign("The expression contained an invalid back reference.");
			break;
		case std::regex_constants::error_brack:
			errordescription.assign("The expression contained mismatched brackets [ and ].");
			break;
		case std::regex_constants::error_paren:
			errordescription.assign("The expression contained mismatched parentheses ( and ).");
			break;
		case std::regex_constants::error_brace:
			errordescription.assign("The expression contained mismatched braces { and }.");
			break;
		case std::regex_constants::error_badbrace:
			errordescription.assign("The expression contained an invalid range between braces { and }.");
			break;
		case std::regex_constants::error_range:
			errordescription.assign("The expression contained an invalid character range.");
			break;
		case std::regex_constants::error_space:
			errordescription.assign("There was insufficient memory to convert the expression into a finite state machine.");
			break;
		case std::regex_constants::error_badrepeat:
			errordescription.assign("The expression contained a repeat specifier (one of *?+{) that was not preceded by a valid regular expression.");
			break;
		case std::regex_constants::error_complexity:
			errordescription.assign("The complexity of an attempted match against a regular expression exceeded a pre-set level.");
			break;
		case std::regex_constants::error_stack:
			errordescription.assign("There was insufficient memory to determine whether the regular expression could match the specified character sequence.");
			break;
		default:
			errordescription.assign("Unknown error in regex.");
			break;
		}

		return false;
	}
	catch (...) {
		errordescription.assign("Unknown error in checkregex.");
		return false;
	}

	return true;
}

void Log::log(const std::string &str, MsgType type)
{
	switch (type)
	{
		case MsgType::INFO:
			(*os) << "Info: ";
			break;
	
		case MsgType::WARNING:
			(*os) << "Warning: ";
			break;

		case MsgType::ERROR:
			(*os) << "Error: ";
			break;

		default:
			break;
	}
	(*os) << str << "\n";
}