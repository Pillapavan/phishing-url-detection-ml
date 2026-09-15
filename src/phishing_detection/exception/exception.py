import sys

class CustomException(Exception):

    def __init__(self,error_message,error_details: sys):
        super().__init__(error_message)
        self.error_message = self.get_detailed_error_message(error_message,error_details)


    @staticmethod
    def get_detailed_error_message(error_message, error_detail: sys):

        _, _, exc_tb = error_detail.exc_info()

        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno

        return (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{line_number}] "
            f"error message [{error_message}]"
        )

    def __str__(self):
        return self.error_message


try:
    x = 10 / 0

except Exception as e:
    raise CustomException(e, sys)
    