using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("order_details")]
    public class OrderDetail
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("order_id")]
        public int? order_id { get; set; }

        [Column("product_id")]
        public int? product_id { get; set; }

        [Column("quantity")]
        public int? quantity { get; set; }

        [Column("unit_price")]
        public decimal? unit_price { get; set; }

        [Column("total_price")]
        public decimal? total_price { get; set; }
    }
}
