# Federared Random Transfer Learning

Để giải quyết vấn đề mất cân bằng phân phối dữ liệu non-iid của dữ liệu, có thể áp dụng phương pháp transfer learning vào trong quá trình huấn luyện.

Phuong pháp gồm có 3 bước được thực hiện trong mỗi round toàn cầu như sau:

- Đầu tiên các client được chọn trong vòng huấn luyện sẽ huấn luyện như Federared Learning bình thường cà gửi mô hình huấn luyện lên trên server tập chung
- Các mô hình sau khi được gửi lên server sẽ được trộn lên và gửi về các client bất kì
- Các client sau khi được gửi mô hình của mình sẽ thực hiện huấn luyện trên bộ dữ liệu của bản thân rồi gửi mô hình lại về server
- Sau khi các client đã gửi mô hình về lần thứ 2, server sẽ bắt đầu thực hiện tổng hợp mô hình

Nhận xét:
- Có tiềm năng nếu giải quyết được vấn đề chọn client để gửi một mô hình đến thay vì random
