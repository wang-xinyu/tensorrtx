/**
 * @class myLogger
 * @brief Custom TensorRT logger for controlling and formatting log output.
 *
 * This class inherits from nvinfer1::ILogger and overrides the log() method
 * to filter and print TensorRT log messages based on a specified minimum
 * severity level.
 */
class myLogger : public nvinfer1::ILogger {
   public:
    /**
     * @brief Minimum severity level to be reported.
     *
     * Only messages with severity <= reportableSeverity will be printed.
     */
    Severity reportableSeverity;

    /**
     * @brief Construct a new myLogger object.
     *
     * @param severity The minimum severity level to report.
     *                 Defaults to Severity::kVERBOSE (log everything).
     */
    myLogger(Severity severity = Severity::kVERBOSE) : reportableSeverity(severity) {}

    /**
     * @brief Callback function invoked by TensorRT to output log messages.
     *
     * The message is printed only if its severity is within the reportable range.
     * Log messages are formatted and written to stdout.
     *
     * @param severity The severity level of the log message.
     * @param msg The log message provided by TensorRT.
     */
    void log(Severity severity, const char* msg) noexcept override {
        if (severity > reportableSeverity)
            return;

        const char* level = "";
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                level = "INTERNAL_ERROR";
                break;
            case Severity::kERROR:
                level = "ERROR";
                break;
            case Severity::kWARNING:
                level = "WARNING";
                break;
            case Severity::kINFO:
                level = "INFO";
                break;
            case Severity::kVERBOSE:
                level = "VERBOSE";
                break;
        }
        std::cout << "[TRT][" << level << "] " << msg << std::endl;
    }
};
