#include "iofiles.h"

#include <list>
#include <regex>

#include <boost/filesystem.hpp>


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
bool iofiles::addpath(
	const boost::filesystem::path &ipath,
	const boost::filesystem::path &opath,
	const std::string regex,
	bool output_must_be_directory) noexcept
{
	using namespace boost::filesystem;


	//status: follow symbol links
	//symlink_status: do not follow symbol links
	file_type itype = status(ipath).type();
	file_type otype = status(opath).type();

	try{
		if (itype == file_not_found) {
			return false;
		}
		else if (itype == directory_file) {
			_iofolder iofolder;
			iofolder.ifolder = directory_iterator(ipath);
			if (regex.length() != 0) {
				iofolder.has_regex = true;
				iofolder.regex.assign(regex);
			}
			else {
				iofolder.has_regex = false;
			}

			if (otype == file_not_found) {
				create_directory(opath);

				iofolder.ofolder = opath;
				folders.push_back(std::move(iofolder));
				return true;
			}
			else if (otype == directory_file) {
				iofolder.ofolder = opath;
				folders.push_back(std::move(iofolder));
				return true;
			}
			else {
				return false;
			}

		}
		else if (itype != status_error) {
			_iofile iofile;
			iofile.ifile = ipath;

			if (otype == file_not_found) {
				char trailing_char = opath.native().back();
				if (trailing_char == '\\' || trailing_char == '/') {    //opath is a non-existing directory
					create_directory(opath);
				}
				else {                                                  //opath is a non-existing file
					if (output_must_be_directory) {
						return false;
					}
				}


				iofile.ofile = opath;
				files.push_back(std::move(iofile));
				return true;
			}
			else if (otype != status_error) {
				iofile.ofile = opath;
				files.push_back(std::move(iofile));
				return true;
			}
			else {
				return false;
			}

		}
		else {
			return false;
		}
	}
	catch(...){
		return false;
	}

}

// output:
//    ifilepath: input file path, which will always be a file.
//    opath: output path, which will be either a file or a directory.
// return value:
//    return false if there is no more file.
bool iofiles::getnextfile(boost::filesystem::path &ifilepath, boost::filesystem::path &opath)
{
	using namespace boost::filesystem;

	if (!files.empty()) {      // if "files" is not empty
		auto i = files.begin();
		ifilepath = std::move((*i).ifile);
		opath = std::move((*i).ofile);
		files.erase(i);
		return true;
	}

	while (!folders.empty()) {   // if "folders" is not empty
		auto i = folders.begin();
		std::regex &regex = (*i).regex;
		bool has_regex = (*i).has_regex;

		while (1) {
			directory_iterator &subfile = (*i).ifolder;


			if (subfile == directory_iterator()/*end iterator*/) {
				break;
			}

			path subfilepath = (*subfile).path();
			//status: follow symbol links
			//symlink_status: do not follow symbol links
			file_type subfiletype = status(subfilepath).type();

			if (subfiletype == directory_file || subfiletype == status_error) {
				subfile++;
				continue;
			}

			if (has_regex) {
				if (!std::regex_match(subfilepath.filename().c_str(), regex)) {
					subfile++;
					continue;
				}
			}
			

			ifilepath = subfilepath;
			opath = (*i).ofolder;
			subfile++;
			return true;
		}

		folders.erase(i);
		continue;

	}


	return false;

}
