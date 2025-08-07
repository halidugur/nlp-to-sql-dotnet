using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace stajprojem.Models
{
    [Table("categories")]
    public class Category
    {
        [Key]
        [Column("id")]
        public int id { get; set; }

        [Column("category_name")]
        public string? category_name { get; set; }

        [Column("description")]
        public string? description { get; set; }

        [Column("created_date")]
        public DateTime? created_date { get; set; }
    }
}
