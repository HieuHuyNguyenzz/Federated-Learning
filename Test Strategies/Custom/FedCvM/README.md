# Federated Cram von Misses 

## Nhận thấy:

- Việc đánh trọng số của mô hình trong quá trình tổng hợp dựa trên một độ đo mức độ non-iid của đữ liệu chỉ hiệu quả trong những round đầu
- Mặt khác càng về các round sau mức độ ảnh hưởng của việc đánh trọng số lên độ chính xác của mô hình tổng hợp so với khi sử dụng FedAvg là không đáng kể

⇒ Thay vì sử dụng chỉ số khác mà trọng số sau tính toán nằm trong một khoảng tương đối nhỏ nào đấy, chúng ta có thể tìm kiếm một độ đo khác giúp làm tăng miền giá trị của trọng số và thực hiện giảm dần miền này sau mỗi round.

## Cram von Misses:

Cramer-von Mises là một phương pháp thống kê được sử dụng để kiểm tra xem một mẫu dữ liệu có tuân theo một phân phối xác định hay không. Phương pháp này là một trong những kiểm định thống kê không tham số, nghĩa là nó không yêu cầu giả định về hình dạng cụ thể của phân phối.

Cramer-von Mises kiểm tra giả thuyết rằng mẫu dữ liệu được kiểm tra có phân phối giống với một phân phối cụ thể (thường là phân phối chuẩn). Nói cách khác, nếu giá trị p (giá trị p-value) của kiểm định này là nhỏ, chúng ta có đủ bằng chứng để bác bỏ giả thuyết rằng mẫu không tuân theo phân phối mong muốn.

Phương pháp Cramer-von Mises được sử dụng trong thống kê, khoa học dữ liệu và các lĩnh vực khác để đánh giá tính phân phối của mẫu dữ liệu.

$$W^2 = (1 / (12n)) + Σ [F_n(X_(i)) - F(X_(i))]^2$$


## Vấn đề:
- Chưa chuẩn hóa được trọng số
- Không có gì đột phá
