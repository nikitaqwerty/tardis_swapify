#include "face_seg/face_seg.h"
#include <exception>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>  // debug

using namespace caffe;

namespace face_seg
{

	FaceSeg::FaceSeg(const string& deploy_file, const string& model_file,
		bool with_gpu, int gpu_device_id) :
		m_num_channels(0), m_with_gpu(with_gpu)
	{
		if (with_gpu)
		{
			Caffe::SetDevice(gpu_device_id);
			Caffe::set_mode(Caffe::GPU);
		}
		else Caffe::set_mode(Caffe::CPU);

		// Load the network
		m_net.reset(new Net<float>(deploy_file, caffe::TEST));
		m_net->CopyTrainedLayersFrom(model_file);

		CHECK_EQ(m_net->num_inputs(), 1) << "Network should have exactly one input.";
		CHECK_EQ(m_net->num_outputs(), 1) << "Network should have exactly one output.";

		Blob<float>* input_layer = m_net->input_blobs()[0];
		m_num_channels = input_layer->channels();
		CHECK(m_num_channels == 3 || m_num_channels == 1)
			<< "Input layer should have 1 or 3 channels.";
		m_input_size = cv::Size(input_layer->width(), input_layer->height());
	}


	FaceSeg::~FaceSeg()
	{
	}

	cv::Mat FaceSeg::process(const cv::Mat& img)
	{
		// Prepare input data
		std::vector<cv::Mat> input_channels;
		wrapInputLayer(input_channels);
		preprocess(img, input_channels);

		// Forward pass
		m_net->Forward();
		
		// Extract background and foreground from output layer
		Blob<float>* output_layer = m_net->output_blobs()[0];
		cv::Mat background(output_layer->height(), output_layer->width(), CV_32F, 
			(void*)output_layer->cpu_data());
		cv::Mat foreground(output_layer->height(), output_layer->width(), CV_32F,
			(void*)(output_layer->cpu_data() + 15*(output_layer->height() * output_layer->width())));
		
		// Calculate argmax
		cv::Mat seg(output_layer->height(), output_layer->width(), CV_8U);
		unsigned char* seg_data = seg.data;
		float* back_data = (float*)background.data;
		float* fore_data = (float*)foreground.data;
		for (int i = 0; i < seg.total(); ++i)
			*seg_data++ = (*back_data++ < *fore_data++) ? 255 : 0;

		// Resize to original image size
		if (seg.size() != img.size())
			cv::resize(seg, seg, img.size(), 0, 0, cv::INTER_NEAREST);

		// Output results
		return seg;
	}

void FaceSeg::wrapInputLayer(std::vector<cv::Mat>& input_channels)
{
    Blob<float>* input_layer = m_net->input_blobs()[0];

    int width = input_layer->width();
    int height = input_layer->height();

	float* input_data = input_layer->mutable_cpu_data();

    for (int i = 0; i < input_layer->channels(); ++i) {
        cv::Mat channel(height, width, CV_32FC1, input_data);
        input_channels.push_back(channel);
        input_data += width * height;
    }
}

void FaceSeg::preprocess(const cv::Mat& img, std::vector<cv::Mat>& input_channels)
{
    // Convert the input image to the input image format of the network.
    cv::Mat sample;
    if (img.channels() == 3 && m_num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
    else if (img.channels() == 4 && m_num_channels == 1)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
    else if (img.channels() == 4 && m_num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
    else if (img.channels() == 1 && m_num_channels == 3)
        cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
    else
        sample = img;

    cv::Mat sample_resized;
    if (sample.size() != m_input_size)
        cv::resize(sample, sample_resized, m_input_size, 0, 0, cv::INTER_CUBIC);
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (m_num_channels == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);

    // Normalize the image by substracting the mean
    subtractMean_c3(sample_float);

	// This operation will write the separate BGR planes directly to the
	// input layer of the network because it is wrapped by the cv::Mat
	// objects in input_channels. 
	cv::split(sample_float, input_channels);
}

void FaceSeg::subtractMean_c3(cv::Mat& img)
{
    int r, c;
    float* img_data = (float*)img.data;
    for(r = 0; r < img.rows; ++r)
    {
        for(c = 0; c < img.cols; ++c)
        {
            *img_data++ -= MB;
            *img_data++ -= MG;
            *img_data++ -= MR;
        }
    }
}

}   // namespace face_seg
