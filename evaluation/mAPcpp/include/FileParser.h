//
// Created by redwan on 12/5/21.
//

#ifndef MAP_FILEPARSER_H
#define MAP_FILEPARSER_H

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/algorithm/string.hpp>
#include <boost/filesystem.hpp>
#include <cassert>
#include <future>
#include <boost/lexical_cast.hpp>
using namespace std;
namespace fs = boost::filesystem;

struct Object{
    /* @brief each object is comprised with 4 parameters
     * @param filename represents which image this object is from
     * @param category represents what is this object, e.g. person, car, traffic light
     * @param box represents bounding box in cv2 rectangle format, i.e., left top right bottom
     */
    string filename;
    string category;
    double conf_score;
    array<double, 4> box;

    friend ostream &operator<<(ostream &os, const Object &object) {
        os << "filename: " << object.filename << " category: " << object.category << " conf_score: "
           << object.conf_score << " box: " ;
        for (int i = 0; i < 4; ++i) {
           os << object.box[i] << " ";
        }
        return os;
    }

};


class FileParser{
public:
    /* @brief this class will parse files from a given directory and generate object that is easy to work with
     * @param dirname: directory where detection or ground truth files are located
     * @param ext: extension of intended files
     */
    FileParser(const string& dirname, const string& ext): dir_(dirname), ext_(ext)
    {

        assert(fs::exists(dirname)  && "directory does not exist");
    }

    virtual ~FileParser()
    {

    }
    /* @brief this function will read all the files asynchronously
     * then generate objects by parsing each files
     */
    vector<shared_ptr<Object>> read()
    {
        vector<shared_ptr<Object>> results;
        vector<future<vector<shared_ptr<Object>>>>parallel;
        auto paths = get_all(dir_, ext_);
        for(auto &path : paths)
        {
            auto data = read_file(path.string());
            parallel.emplace_back(async(parse_txt_string, data, basename(path)));

        }

        for(auto & i : parallel) {
            auto objects = i.get();
            for (auto& obj : objects) {
                results.emplace_back(obj);
            }
        }


        return results;
    }

protected:
    /* @breif this function will read all lines from the file
     * @pram filename: full path of txt file
     * @return: content of the file in string format
     */
    string read_file(const string& filename)
    {
        string data = "", line;
        ifstream File(filename);
        if (File.is_open())
        {
            while ( getline(File, line) )
            {

                if (line.length())
                {
                    data += line + "\n";
                }
            }
            File.close();
        }
        boost::trim(data);
        return data;
    }

    /**
     * @brief   Return the filenames of all files that have the specified extension
     *          in the specified directory and all subdirectories.
     */
    std::vector<fs::path> get_all(fs::path const & root, std::string const & ext)
    {
        std::vector<fs::path> paths;

        if (fs::exists(root) && fs::is_directory(root))
        {
            for (auto const & entry : fs::recursive_directory_iterator(root))
            {
                if (fs::is_regular_file(entry) && entry.path().extension() == ext)
                    paths.emplace_back(entry.path());
            }
        }

        return paths;
    }

private:
    string dir_, ext_;

protected:
    /* @brief this function will parse the string data from the file and return a vector of objects
     * @param data: string data from read_file() function
     * @param filename: a file identifier for each object
     */
    static vector<shared_ptr<Object>> parse_txt_string(const string& data, const string& filename)
    {
        vector<string> lines;
        vector<shared_ptr<Object>> results;

        // convert the long string data to a vector of lines based on new line delimiters
        boost::split(lines,data,boost::is_any_of("\n"));

        for (auto line: lines)
        {
            boost::trim(line);
            vector<string> words;

            // convert each line to a vector of words based on space delimiter
            boost::split(words, line, boost::is_any_of(" "));

            // in the ground truth line there is no confidence score, so we are anticipating a vector of size 5
            // on the other hand, detection line consists of 6 words, i.e., label, conf, left, top, right, bottom
            if (words.size() == 5 || words.size() == 6)
            {
                // use shared memory for efficient storage
                auto obj = make_shared<Object>();

                obj->category = words[0];
                obj->filename = filename;
                boost::trim(obj->filename);

                int startIndex, boxIndex = 0;

                // if there is no confidence score use the -1 for ground truth
                if (words.size() == 5)
                {
                    obj->conf_score = -1.0;
                    startIndex = 1;
                }
                else
                {
                    startIndex = 2;
                    auto value = words[1];
                    boost::trim(value);
                    obj->conf_score = boost::lexical_cast<double>(value);
                }

                // If conf exists in words we need to skip first two index else skip one index, i.e., label only
                while (startIndex < words.size()) {
                    auto value = words[startIndex++];
                    boost::trim(value);
                    obj->box[boxIndex++] = boost::lexical_cast<double>(value);
                }
                results.emplace_back(obj);
            }

        }
        return results;
    }
};
#endif //MAP_FILEPARSER_H
