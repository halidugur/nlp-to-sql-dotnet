namespace stajprojem.Models
{
    public class ProductFilterViewModel
    {
        public int? CategoryId { get; set; }
        public string? ProductName { get; set; }
        public decimal? MinPrice { get; set; }
        public decimal? MaxPrice { get; set; }
        public int? MinStock { get; set; }
        public int? MaxStock { get; set; }
    }
}
