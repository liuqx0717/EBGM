#pragma once

#include <string>
#include <regex>

#include <boost/filesystem.hpp>

// won't enum subdirectories
// won't cache subfile names to memory, that is, you can use this class to enum a directory with tons of subfiles.
// no guard for multithreading
class iofiles {
private:
	struct _iofile {
		boost::filesystem::path ifile;
		boost::filesystem::path ofile;
	};

	struct _iofolder {
		boost::filesystem::directory_iterator ifolder;
		bool has_regex;
		std::regex regex;
		boost::filesystem::path ofolder;
	};

	std::list<_iofile> files;
	std::list<_iofolder> folders;

public:
	iofiles() {  }

	// ipath must exist, otherwise this function will return false.
	// when ipath is a directory, you can use regex as a filter of the subfiles
	// if ipath is a file, opath can be one of the following:
	//    an existing file name
	//    an existing directory name
	//    a non-existing file name
	//    a non-existing directory name with a trailing "/" or "\"
	// if ipath is a directory, opath can be either of the following:
	//    an existing directory name
	//    a non-existing directory name
	// this function will create the non-existing directory, if that fails, this function will return false.
	// if an error occurred (such as a regex error), this function will return false.
	bool addpath(
		const boost::filesystem::path &ipath,
		const boost::filesystem::path &opath,
		const std::string regex = std::string(),
		bool output_must_be_directory = false) noexcept;

	// output:
	//    ifilepath: input file path, which will always be a file.
	//    opath: output path, which will be either a file or a directory.
	// return value:
	//    return false if there is no more file.
	bool getnextfile(boost::filesystem::path &ifilepath, boost::filesystem::path &opath);
	void clear() { files.clear(); folders.clear(); }
	bool empty() const {return files.empty() && folders.empty();}
};

