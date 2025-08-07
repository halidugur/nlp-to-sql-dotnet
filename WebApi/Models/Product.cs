using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("products")]
    public class Product
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("product_name")]
        public string? product_name { get; set; }

        [Column("category_id")]
        public int? category_id { get; set; }

        [Column("supplier_id")]
        public int? supplier_id { get; set; }

        [Column("unit_price")]
        public decimal? unit_price { get; set; }

        [Column("unit")]
        public string? unit { get; set; }

        [Column("stock_quantity")]
        public int? stock_quantity { get; set; }

        [Column("description")]
        public string? description { get; set; }

        [Column("created_date")]
        public DateTime? created_date { get; set; }
    }
}
