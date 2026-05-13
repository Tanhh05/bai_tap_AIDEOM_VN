# Bài 2 - Phân bổ ngân sách đầu tư số

Kết quả chính sau khi chạy `python3 src/optimization/ex2_budget_lp.py`:

- Phân bổ tối ưu cơ sở: hạ tầng số 25, AI và dữ liệu 15, nhân lực số 20, R&D 40 nghìn tỷ VND.
- Giá trị tối ưu `Z* = 112.25` nghìn tỷ VND GDP kỳ vọng.
- SciPy `linprog` và PuLP cho cùng nghiệm tối ưu.
- Shadow price của ràng buộc ngân sách tổng là 1.35. Trong vùng nghiệm hiện tại, thêm 1 nghìn tỷ VND ngân sách làm GDP kỳ vọng tăng khoảng 1.35 nghìn tỷ VND.
- Khi ngân sách tăng lên 120 và 140, nghiệm dồn phần tăng thêm vào R&D, lần lượt cho `Z* = 139.25` và `166.25`.
- Kịch bản ưu tiên nhân lực số `x3 >= 30` vẫn khả thi, nhưng `Z*` giảm còn 108.25 do phải chuyển 10 nghìn tỷ VND từ R&D sang nhân lực số.
