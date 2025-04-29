import java.util.ArrayList;

public class EjemploWildCards {
	
	public static double suma(ArrayList<? extends Number> a){
		double aux=0.0;
		for(int i=0;i<a.size();i++){
			aux+=a.get(i).doubleValue();
		}
		Number n=a.get(0);
		return aux;
	}

	public static void imprime(ArrayList<?> lista){
		for(int i=0;i<lista.size();i++){
		System.out.print(lista.get(i)+",");
		}
		System.out.println();
		Object o=lista.get(0);
	}
	public static void main(String[] args) {
		ArrayList<Integer> numeros=new ArrayList<>();
		numeros.add(10);
		numeros.add(20);
		numeros.add(5);
		numeros.add(4);
		numeros.add(9);
		System.out.println(EjemploWildCards.suma(numeros));
		imprime(numeros);
	}
}
